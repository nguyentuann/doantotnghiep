# ============================================================
# Frontend Dockerfile — React + Vite → Nginx (multi-stage)
# ------------------------------------------------------------
# Build context = thư mục `frontend/`
#   docker build -t typhoon-frontend:latest ./frontend
# ============================================================

# --- Stage 1: Build React app ---
FROM node:20-alpine AS build
WORKDIR /app

# Cài deps (cache theo package-lock)
COPY package.json package-lock.json ./
RUN npm ci

# Build production bundle → /app/dist
COPY . .
RUN npm run build

# --- Stage 2: Serve bằng Nginx ---
FROM nginx:1.27-alpine

# Cấu hình SPA + proxy /api → backend
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Static files
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
