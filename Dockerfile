FROM node:20-alpine

# Install Python and dependencies only if needed for the build process
RUN apk add --no-cache python3 py3-pip

# Set working directory
WORKDIR /app

# Copy package files first to leverage Docker layer caching
COPY package*.json ./

# Install Node.js dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Build the Docusaurus site
RUN npm run build

# Install serve globally to serve the static files
RUN npm install -g serve

# Set default port and expose it
ENV PORT=3000
EXPOSE 3000

# Start serving the built static site
CMD ["serve", "-s", "build", "--listen", "3000"]