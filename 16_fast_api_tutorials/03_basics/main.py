from fastapi import FastAPI

app = FastAPI() # Create an instances of FastAPI

# Define a route
@app.get("/")
async def root():
    return {"message": "Hello World we are learning on codanics.com, HI HOW  what ARE YOY"}
