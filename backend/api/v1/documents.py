from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from core.config import settings
from ml_pipeline.engine import IntelligentDocumentProcessor
from api.dependencies import get_db
from database.repository import DocumentRepository
import aiofiles
import os
import uuid
import io
import pandas as pd

router = APIRouter()

# Load the ML engine directly into the API memory (Bypassing Celery/Redis)
print("Loading ML Models directly into FastAPI...")
ocr_engine = IntelligentDocumentProcessor()

def parse_gemini_error(error_content: str) -> tuple:
    try:
        import json
        err_data = json.loads(error_content)
        if "error" in err_data:
            err = err_data["error"]
            code = err.get("code", 500)
            msg = err.get("message", "Unknown Gemini API error")
            
            # Map code/status
            if err.get("status") == "RESOURCE_EXHAUSTED" or code == 429:
                return 429, f"Gemini API Quota Exceeded: {msg}"
            return code if isinstance(code, int) else 500, msg
    except Exception:
        pass
    
    # Fallback to string search
    if "RESOURCE_EXHAUSTED" in error_content or "429" in error_content:
        return 429, "Gemini API rate limit or quota exceeded. Please try again later."
    if "503" in error_content or "UNAVAILABLE" in error_content:
        return 503, "Gemini API service is temporarily unavailable. Please retry shortly."
        
    return 500, f"AI Extraction Pipeline Error: {error_content}"

def merge_page_data(existing_data: dict, new_page_data: dict) -> dict:
    merged = existing_data.copy()
    for key, val in new_page_data.items():
        if isinstance(val, dict):
            if key not in merged or not isinstance(merged[key], dict):
                merged[key] = val.copy()
            else:
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, list):
                        if sub_key not in merged[key] or not isinstance(merged[key][sub_key], list):
                            merged[key][sub_key] = sub_val.copy()
                        else:
                            merged[key][sub_key].extend(sub_val)
                    else:
                        merged[key][sub_key] = sub_val
        elif isinstance(val, list):
            if key not in merged or not isinstance(merged[key], list):
                merged[key] = val.copy()
            else:
                merged[key].extend(val)
        else:
            merged[key] = val
    return merged

def get_val_global(d: dict, target_key: str):
    if not d:
        return None
    norm_target = "".join(c.lower() for c in target_key if c.isalnum())
    for k, v in d.items():
        if "".join(c.lower() for c in k if c.isalnum()) == norm_target:
            return v
    return None

def normalize_batch_summary(batch_list: list) -> list:
    normalized = []
    if not batch_list or not isinstance(batch_list, list):
        return normalized
    for row in batch_list:
        if not isinstance(row, dict):
            continue
        normalized.append({
            "p_order": get_val_global(row, "p_order") or get_val_global(row, "p.order") or get_val_global(row, "porder") or get_val_global(row, "order"),
            "material_code": get_val_global(row, "material_code") or get_val_global(row, "materialcode") or get_val_global(row, "code"),
            "material_description": get_val_global(row, "material_description") or get_val_global(row, "description") or get_val_global(row, "materialdesc") or get_val_global(row, "materialdescription"),
            "batch_no": get_val_global(row, "batch_no") or get_val_global(row, "batch") or get_val_global(row, "batchno"),
            "t_qty": get_val_global(row, "t_qty") or get_val_global(row, "t.qty") or get_val_global(row, "totalqty") or get_val_global(row, "qty"),
            "unit": get_val_global(row, "unit"),
            "b_qty": get_val_global(row, "b_qty") or get_val_global(row, "b.qty") or get_val_global(row, "balanceqty") or get_val_global(row, "batchqty") or get_val_global(row, "bqty"),
            "t_c_wt": get_val_global(row, "t_c_wt") or get_val_global(row, "t.c.wt") or get_val_global(row, "totalcastwt") or get_val_global(row, "castweight") or get_val_global(row, "tcwt"),
            "s_order": get_val_global(row, "s_order") or get_val_global(row, "s.order") or get_val_global(row, "salesorder") or get_val_global(row, "saleorder"),
            "s_item": get_val_global(row, "s_item") or get_val_global(row, "s.item") or get_val_global(row, "salesitem"),
            "c_code": get_val_global(row, "c_code") or get_val_global(row, "c.code") or get_val_global(row, "customercode"),
            "division": get_val_global(row, "division") or get_val_global(row, "div")
        })
    return normalized

def map_dynamic_to_queue_page(extracted_data: dict, page_num: int) -> dict:
    metadata = extracted_data.get("document_metadata", {}) or {}
    product = extracted_data.get("product_details", {}) or {}
    pouring = extracted_data.get("pouring_details", {}) or {}
    inspection = extracted_data.get("inspection_parameters", {}) or {}
    tables = extracted_data.get("tables", {}) or {}
    signatures = extracted_data.get("signatures", {}) or {}

    get_val = get_val_global

    prod_plan = {
        "heat_no": get_val(metadata, "heat_no") or get_val(metadata, "cycle_no"),
        "planning_date": get_val(metadata, "date"),
        "pouring_date": get_val(pouring, "pouring_date") or get_val(pouring, "date"),
        "customer": get_val(product, "customer"),
        "grade": get_val(product, "grade"),
        "casting_weight": get_val(product, "casting_weight"),
        "liquid_weight": get_val(product, "liquid_weight"),
        "qty": get_val(product, "qty") or get_val(product, "quantity"),
        "sample_bulk": get_val(product, "sample_bulk") or get_val(product, "sample_/_bulk"),
        "finish_type": get_val(product, "finish_type"),
        "pattern_code": get_val(product, "pattern_code"),
        "pattern_serial_no": get_val(product, "pattern_serial_no"),
        "pattern_type": get_val(product, "pattern_type"),
        "drawing_number": get_val(product, "drawing_number") or get_val(product, "drawing_no"),
        "part_no": get_val(product, "part_no"),
        "pcs_in_box": get_val(product, "pcs_in_box"),
        "no_of_core_boxes": get_val(product, "no_of_core_boxes"),
        "no_of_cores": get_val(product, "no_of_cores") or get_val(product, "no._of_cores"),
        "method_remarks": get_val(product, "method_remarks"),
    }

    pour_details = {
        "pouring_date": get_val(pouring, "pouring_date") or get_val(pouring, "date"),
        "pouring_time": get_val(pouring, "pouring_time") or get_val(pouring, "time"),
        "pouring_qty": get_val(pouring, "pouring_qty"),
        "pouring_sec": get_val(pouring, "pouring_sec") or get_val(pouring, "duration") or get_val(pouring, "pouring_time"),
        "tapping_temp": get_val(pouring, "tapping_temp") or get_val(pouring, "tapping_temperature"),
        "pouring_temp": get_val(pouring, "pouring_temp") or get_val(pouring, "pouring_temperature"),
        "laddle_temp": get_val(pouring, "laddle_temp") or get_val(pouring, "ladle_temp"),
        "pouring_weight": get_val(pouring, "pouring_weight") or get_val(product, "liquid_weight"),
        "core_making": get_val(pouring, "core_making"),
    }

    qa_params = {
        "hardness_mould": get_val(inspection, "hardness_range_mould") or get_val(inspection, "mould_hardness_range") or get_val(inspection, "hardness_range_mould_70_to_85"),
        "hardness_core": get_val(inspection, "hardness_range_core") or get_val(inspection, "core_hardness_range") or get_val(inspection, "hardness_range_core_65_to_85") or get_val(inspection, "hardness/range(core)"),
        "coating_baume_value": get_val(inspection, "coating_baume_value") or get_val(inspection, "coating_baume_value_range") or get_val(inspection, "coating_baume_value_range_53_to_65"),
        "core_oven_baking_on_time": get_val(inspection, "core_oven_baking_on_time"),
        "core_oven_baking_off_time": get_val(inspection, "core_oven_baking_off_time"),
        "core_oven_preheating_temp": get_val(inspection, "core_oven_preheating_temp"),
        "no_of_cores": get_val(inspection, "no_of_cores"),
        "mould_coating": get_val(inspection, "mould_coating"),
        "core_coating": get_val(inspection, "core_coating"),
        "lettering_checking": get_val(inspection, "lettering_checking"),
        "mould_core_visual_checking": get_val(inspection, "mould_core_visual_checking") or get_val(inspection, "mould_&_core_visual_checking"),
        "mould_core_coating_application": get_val(inspection, "mould_core_coating_application") or get_val(inspection, "mould_&_core_coating_application"),
        "core_setting_wall_thickness": get_val(inspection, "core_setting_wall_thickness"),
        "mould_core_preheating": get_val(inspection, "mould_core_preheating") or get_val(inspection, "mould_&_core_preheating"),
        "templates_checking": get_val(inspection, "templates_checking"),
        "core_setting_inspector": get_val(inspection, "core_setting_inspector") or get_val(inspection, "core_setting"),
        "closing_inspector": get_val(inspection, "closing_inspector") or get_val(inspection, "closing"),
        "pouring_inspector": get_val(inspection, "pouring_inspector") or get_val(inspection, "pouring"),
    }

    sleeve_list = []
    for s in tables.get("sleeves", []) or []:
        sleeve_list.append({
            "sle_code": get_val(s, "code"),
            "sle_name": get_val(s, "name"),
            "slv_qty": get_val(s, "qty")
        })

    consumable_list = []
    for c in tables.get("consumables", []) or []:
        consumable_list.append({
            "item": get_val(c, "item"),
            "quantity": get_val(c, "qty") or get_val(c, "quantity")
        })

    headers = {
        "form_id": get_val(metadata, "form_id"),
        "heat_no": get_val(metadata, "heat_no") or get_val(metadata, "cycle_no"),
        "planning_date": get_val(metadata, "date"),
        "pouring_date_header": get_val(pouring, "pouring_date") or get_val(pouring, "date"),
    }

    return {
        "page_number": page_num,
        "document_headers": headers,
        "production_plan": prod_plan,
        "pouring_details": pour_details,
        "qa_parameters": qa_params,
        "bottom_signatures": signatures,
        "sleeve_table": sleeve_list,
        "handwritten_consumables_list": consumable_list
    }

@router.post("/process")
async def upload_and_process_document(
    file: UploadFile = File(None),
    filename: str = None,
    page: int = 0,
    task_id: str = None,
    db = Depends(get_db)
):
    """
    Accepts an industrial scan page-by-page. For initial page (page=0), accepts a file upload.
    For subsequent pages, accepts filename to reuse the saved document path.
    Integrates results incrementally into MongoDB.
    """
    if not file and not filename:
        raise HTTPException(status_code=400, detail="Either file or filename must be provided.")

    if file:
        allowed_types = ["image/jpeg", "image/png", "application/pdf"]
        file_extension = file.filename.split(".")[-1].lower()
        
        # Verify content type or extension matches allowed files (more robust for browser variations)
        is_allowed = (
            file.content_type in allowed_types or 
            file_extension in ["pdf", "jpg", "jpeg", "png"]
        )
        if not is_allowed:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use JPG, PNG, or PDF.")

        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)

        # Save file
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
            
        filename_used = unique_filename
    else:
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Saved file {filename} not found on server.")
        filename_used = filename

    try:
        # Process the specific page
        result_payload = await run_in_threadpool(ocr_engine.process_document, file_path, page_num=page)
        
        # Enhanced debugging log
        print(f"DEBUG - Extracted page results payload: {result_payload}")
        
        if isinstance(result_payload, dict) and "error" in result_payload:
            error_msg = result_payload['error']
            status, cleaned_msg = parse_gemini_error(error_msg)
            
            raise HTTPException(
                status_code=status, 
                detail=cleaned_msg
            )
            
        extracted_data = result_payload["extracted_data"]
        total_pages = result_payload["total_pages"]
        
        # Normalize/map to structured queue_pages schema
        if "queue_pages" not in extracted_data:
            page_data = map_dynamic_to_queue_page(extracted_data, page_num=page)
            raw_batch = extracted_data.get("tables", {}).get("batch_summary", []) or []
            new_extracted_data = {
                "queue_pages": [page_data],
                "batch_summary": normalize_batch_summary(raw_batch)
            }
        else:
            new_extracted_data = extracted_data
            if "batch_summary" in new_extracted_data:
                new_extracted_data["batch_summary"] = normalize_batch_summary(new_extracted_data["batch_summary"])
            elif "tables" in new_extracted_data and "batch_summary" in new_extracted_data["tables"]:
                new_extracted_data["batch_summary"] = normalize_batch_summary(new_extracted_data["tables"]["batch_summary"])

        # Save to database (MongoDB) and merge if task_id is provided
        repo = DocumentRepository(db)
        if task_id:
            existing_doc = await repo.get_document(task_id)
            if existing_doc:
                existing_extracted = existing_doc.get("extracted_data", {}) or {}
                if "queue_pages" in existing_extracted and "queue_pages" in new_extracted_data:
                    existing_pages = existing_extracted.get("queue_pages", [])
                    new_pages = new_extracted_data.get("queue_pages", [])
                    
                    # Remove any existing page with the same page number and append the new page
                    existing_pages = [p for p in existing_pages if p.get("page_number") != page]
                    existing_pages.extend(new_pages)
                    existing_pages.sort(key=lambda p: p.get("page_number", 0))
                    
                    # Merge batch summary tables (normalizing existing too to cleanse any old schemas)
                    existing_batch = normalize_batch_summary(existing_extracted.get("batch_summary", []) or [])
                    new_batch = new_extracted_data.get("batch_summary", []) or []
                    existing_batch.extend(new_batch)
                    
                    # Deduplicate batch summary entries
                    seen_batch = set()
                    unique_batch = []
                    for row in existing_batch:
                        row_key = (row.get("material_code"), row.get("batch_no"), row.get("t_qty"))
                        if row_key not in seen_batch:
                            seen_batch.add(row_key)
                            unique_batch.append(row)
                    
                    accumulated_data = {
                        "queue_pages": existing_pages,
                        "batch_summary": unique_batch
                    }
                else:
                    accumulated_data = merge_page_data(existing_extracted, new_extracted_data)
                await repo.save_document(task_id, accumulated_data, filename=filename_used)
            else:
                await repo.save_document(task_id, new_extracted_data, filename=filename_used)
                accumulated_data = new_extracted_data
        else:
            task_id = uuid.uuid4().hex
            await repo.save_document(task_id, new_extracted_data, filename=filename_used)
            accumulated_data = new_extracted_data
        
        return {
            "message": "Page processed successfully",
            "filename": filename_used,
            "task_id": task_id,
            "data": accumulated_data,
            "current_page": page,
            "total_pages": total_pages,
            "has_next_page": page < total_pages - 1
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed inside route: {str(e)}")

@router.get("/")
async def get_all_processed_documents(db = Depends(get_db)):
    """
    Retrieves all processed document records from the database.
    """
    try:
        repo = DocumentRepository(db)
        records = await repo.get_all_documents()
        for doc in records:
            if "extracted_data" in doc and doc["extracted_data"]:
                data = doc["extracted_data"]
                if "batch_summary" in data:
                    data["batch_summary"] = normalize_batch_summary(data["batch_summary"])
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve records: {str(e)}")

@router.get("/export")
async def export_all_data_to_excel(db = Depends(get_db)):
    """
    Aggregates all processed document records and converts them to a multi-sheet Excel file.
    Sheet 1: Queue Data (Pages 1-5)
    Sheet 2: Batch Summary (Page 6)
    """
    try:
        repo = DocumentRepository(db)
        documents = await repo.get_all_documents()

        queue_rows = []
        batch_rows = []

        # Parse the JSON structure into flat rows for Excel
        for doc in documents:
            data = doc.get("extracted_data", {})
            
            # 1. Flatten Queue Pages (Original / 6-Page schemas)
            if "queue_pages" in data:
                for page in data.get("queue_pages", []):
                    prod = page.get("production_plan", {}) or {}
                    qa = page.get("qa_parameters", {}) or {}
                    pour = page.get("pouring_details", {}) or {}
                    
                    queue_rows.append({
                        "Task ID": doc.get("task_id", "N/A"),
                        "Page No": page.get("page_number", ""),
                        "Heat No": prod.get("heat_no", ""),
                        "Planning Date": prod.get("planning_date", ""),
                        "Pouring Date": prod.get("pouring_date", ""),
                        "Customer": prod.get("customer", ""),
                        "Grade": prod.get("grade", ""),
                        "Casting Wt": prod.get("casting_weight", ""),
                        "Mould Hardness": qa.get("hardness_mould", ""),
                        "Core Hardness": qa.get("hardness_core", ""),
                        "Pouring Time": pour.get("pouring_time", ""),
                        "Tapping Temp": pour.get("tapping_temp", ""),
                        "Pouring Temp": pour.get("pouring_temp", ""),
                        "Laddle Temp": pour.get("laddle_temp", ""),
                        "Pouring Wt": pour.get("pouring_weight", "")
                    })
            
            # 2. Flatten Dynamic Schema
            elif "document_metadata" in data or "pouring_details" in data:
                metadata = data.get("document_metadata", {}) or {}
                prod = data.get("product_details", {}) or {}
                pour = data.get("pouring_details", {}) or {}
                inspect = data.get("inspection_parameters", {}) or {}
                
                temps_str = pour.get("pouring_temperature", "") or ""
                temps = [t.strip() for t in temps_str.split(',')] if temps_str else [""]
                
                durations_str = pour.get("duration", "") or ""
                durations = [d.strip() for d in durations_str.split(',')] if durations_str else [""]
                
                count = max(len(temps), len(durations), 1)
                
                for i in range(count):
                    p_temp = temps[i] if i < len(temps) else ""
                    p_dur = durations[i] if i < len(durations) else ""
                    
                    queue_rows.append({
                        "Task ID": doc.get("task_id", "N/A"),
                        "Page No": f"Pour {i+1}",
                        "Heat No": metadata.get("heat_no", ""),
                        "Planning Date": metadata.get("date", ""),
                        "Pouring Date": pour.get("date", ""),
                        "Customer": prod.get("customer", ""),
                        "Grade": prod.get("grade", ""),
                        "Casting Wt": prod.get("casting_weight", ""),
                        "Mould Hardness": inspect.get("mould_hardness_range", ""),
                        "Core Hardness": inspect.get("core_hardness_range", ""),
                        "Pouring Time": p_dur,
                        "Tapping Temp": pour.get("tapping_temperature", ""),
                        "Pouring Temp": p_temp,
                        "Ladle Temp": pour.get("laddle_temp", ""),
                        "Pouring Wt": pour.get("pouring_weight", "")
                    })

            # 3. Flatten Batch Summary Table (Original / 6-Page schemas)
            if "batch_summary" in data:
                normalized_batch = normalize_batch_summary(data.get("batch_summary", []))
                for row in normalized_batch:
                    batch_rows.append({
                        "Task ID": doc.get("task_id", "N/A"),
                        "P.Order": row.get("p_order", ""),
                        "Material Code": row.get("material_code", ""),
                        "Material Description": row.get("material_description", ""),
                        "Batch No": row.get("batch_no", ""),
                        "Total Qty": row.get("t_qty", ""),
                        "Unit": row.get("unit", ""),
                        "B.Qty": row.get("b_qty", ""),
                        "T.C.Wt": row.get("t_c_wt", ""),
                        "S.Order": row.get("s_order", ""),
                        "S.Item": row.get("s_item", ""),
                        "C.Code": row.get("c_code", ""),
                        "Division": row.get("division", "")
                    })
            # 4. Flatten Batch Summary Table (Dynamic Schema)
            elif "tables" in data and "batch_summary" in data.get("tables", {}):
                for row in data.get("tables", {}).get("batch_summary", []):
                    normalized_row = normalize_batch_summary([row])[0] if normalize_batch_summary([row]) else {}
                    batch_rows.append({
                        "Task ID": doc.get("task_id", "N/A"),
                        "P.Order": normalized_row.get("p_order", ""),
                        "Material Code": normalized_row.get("material_code", ""),
                        "Material Description": normalized_row.get("material_description", ""),
                        "Batch No": normalized_row.get("batch_no", ""),
                        "Total Qty": normalized_row.get("t_qty", ""),
                        "Unit": normalized_row.get("unit", ""),
                        "B.Qty": normalized_row.get("b_qty", ""),
                        "T.C.Wt": normalized_row.get("t_c_wt", ""),
                        "S.Order": normalized_row.get("s_order", ""),
                        "S.Item": normalized_row.get("s_item", ""),
                        "C.Code": normalized_row.get("c_code", ""),
                        "Division": normalized_row.get("division", "")
                    })

        # Convert to Pandas DataFrames
        df_queue = pd.DataFrame(queue_rows) if queue_rows else pd.DataFrame(columns=["Heat No", "Pouring Date", "Customer"])
        df_batch = pd.DataFrame(batch_rows) if batch_rows else pd.DataFrame(columns=["Material Code", "Batch No", "Total Qty"])
            
        # Write to memory buffer
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_queue.to_excel(writer, index=False, sheet_name='Production Queue (P1-P5)')
            df_batch.to_excel(writer, index=False, sheet_name='Batch Summary (P6)')
            
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=manufacturing_records.xlsx"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")

@router.get("/status/{task_id}")
async def get_processing_status(task_id: str):
    return {"task_id": task_id, "status": "SYNC_MODE_ACTIVE", "message": "Redis is disabled. Check the main /process route for output."}