from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from routes import chat
from core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="RFP Chat API powered by Azure OpenAI.",
    version="1.0.0",
    docs_url="/swagger"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api")

# Custom Swagger UI route
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{settings.PROJECT_NAME} - Swagger UI",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
    )

@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Service is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rfp-chat-agent-api.main:app", host="0.0.0.0", port=8000, reload=True)