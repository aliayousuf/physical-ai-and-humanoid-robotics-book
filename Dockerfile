FROM node:20-alpine

# Install Python and dependencies
RUN apk add --no-cache python3 py3-pip

# Set working directory
WORKDIR /app

# Copy all application code
COPY . .

# Install Node.js dependencies for frontend
RUN npm install && npm run build || echo "Frontend build failed, continuing..."

# Install Python dependencies for backend
RUN if [ -f "backend/requirements.txt" ]; then \
        cd backend && pip install --no-cache-dir --break-system-packages -r requirements.txt; \
    fi

# Expose port (Railway will assign the actual port)
EXPOSE 8000

# Default command that can be overridden by Railway
WORKDIR /app
CMD ["python", "/app/simple_server.py"]