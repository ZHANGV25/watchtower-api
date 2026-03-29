# Dockerfile for Lambda deployment
FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies for OpenCV
RUN yum install -y \
    mesa-libGL \
    glib2 \
    && yum clean all

# Copy requirements and install Python dependencies
# Use Lambda-specific requirements (no uvicorn needed)
COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements-lambda.txt --target ${LAMBDA_TASK_ROOT}

# Copy application code
COPY *.py ${LAMBDA_TASK_ROOT}/
COPY models.py ${LAMBDA_TASK_ROOT}/
COPY routes/ ${LAMBDA_TASK_ROOT}/routes/

# Download YOLO model (if needed)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

# Set Lambda handler
CMD ["lambda_segment.handler"]
