FROM node:18-alpine AS build

WORKDIR /app

ARG REACT_APP_API_URL=http://localhost:4055
ARG REACT_APP_WS_URL=ws://localhost:8055
ARG REACT_APP_MCP_URL=http://localhost:9055

ENV REACT_APP_API_URL=$REACT_APP_API_URL
ENV REACT_APP_WS_URL=$REACT_APP_WS_URL
ENV REACT_APP_MCP_URL=$REACT_APP_MCP_URL

COPY package.json ./
RUN npm install --legacy-peer-deps

COPY . .
ENV DISABLE_ESLINT_PLUGIN=true
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
