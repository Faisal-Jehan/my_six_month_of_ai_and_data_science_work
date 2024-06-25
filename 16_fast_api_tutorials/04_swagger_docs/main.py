from fastapi import FastAPI

app = FastAPI() # Create an instances of FastAPI

# Define a route 
@app.get("/")
async def root():
    return {"message": "Hello World we are learning on codanicsss.com, I m alos have a yt channel"}

# Define a route 
@app.post("/")
async def post_root():
    return {"message": "This is a POST request"}

# Define a route
@app.put ("/{item_id}")
async def put_item(item_id: int):
    return {"message": f"Item ID is {item_id}"}