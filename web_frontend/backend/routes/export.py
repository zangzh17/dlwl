"""
Export API Routes.

Handles exporting optimization results in various formats.
"""

import io
import base64
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal
import numpy as np

from ..services.task_manager import task_manager, TaskStatus


router = APIRouter(prefix="/api", tags=["export"])


class ExportRequest(BaseModel):
    """Request schema for export."""
    format: Literal["csv", "npy", "json"] = Field("csv", description="Export format")
    data_type: Literal["phase", "height", "intensity"] = Field("phase", description="Data to export")


class ExportResponse(BaseModel):
    """Response for export request."""
    success: bool
    filename: Optional[str] = None
    content_type: Optional[str] = None
    data: Optional[str] = None  # Base64 encoded for binary formats
    error: Optional[str] = None


@router.post("/export/{task_id}")
async def export_result(task_id: str, request: ExportRequest):
    """Export optimization result data.

    Supports:
    - CSV: Comma-separated values (text)
    - NPY: NumPy binary format (base64 encoded)
    - JSON: JSON format

    Data types:
    - phase: Optimized phase distribution
    - height: Surface height profile
    - intensity: Simulated intensity pattern
    """
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Task not completed: {task.status.value}")

    if not task.result:
        raise HTTPException(status_code=400, detail="No result available")

    # Get the data array
    result = task.result.get('result', {})

    if request.data_type == 'phase':
        data_key = 'phase'
        filename_base = 'doe_phase'
    elif request.data_type == 'height':
        data_key = 'height'
        filename_base = 'doe_height'
    else:
        data_key = 'simulated_intensity'
        filename_base = 'doe_intensity'

    data = result.get(data_key)
    if data is None:
        raise HTTPException(status_code=400, detail=f"No {request.data_type} data available")

    # Convert to numpy array
    arr = np.array(data)

    if request.format == 'csv':
        # Export as CSV
        output = io.StringIO()
        np.savetxt(output, arr, delimiter=',', fmt='%.8e')
        content = output.getvalue()

        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename="{filename_base}.csv"'
            }
        )

    elif request.format == 'npy':
        # Export as NPY (base64 encoded)
        output = io.BytesIO()
        np.save(output, arr)
        output.seek(0)
        data_base64 = base64.b64encode(output.read()).decode('utf-8')

        return ExportResponse(
            success=True,
            filename=f"{filename_base}.npy",
            content_type="application/octet-stream",
            data=data_base64
        )

    else:  # json
        # Export as JSON
        return {
            'success': True,
            'filename': f"{filename_base}.json",
            'shape': list(arr.shape),
            'data': arr.tolist()
        }
