from src.database.qdrant_controller import QdrantController
from qdrant_client import models

crud = QdrantController()                 
# defaults to localhost ports

# 1. Create a collection
deleted = crud.delete_collection("cities2")

crud.create_collection("cities2", vector_size=4, distance="cosine")

# 2. Insert / update points
points = [
    crud.make_point(1, [0.05, 0.61, 0.76, 0.74], {"city": "Berlin"}),
    crud.make_point(2, [0.19, 0.81, 0.75, 0.11], {"city": "London"}),
]
crud.upsert_points("cities", points)
city_filter = crud.make_filter("city", "Ber")

# 2️⃣  Scroll through matching points (limit = page size)
points, next_page = crud.text_search(
    collection="cities",
    scroll_filter=city_filter,
    limit=100,           # increase or loop with next_page to exhaust results
)
print(points, next_page)
exit()

# 3a. Read back points by id
print(crud.get_points("cities", [1, 2]))

# 3b. Vector search
results = crud.search("cities", [0.05, 0.60, 0.77, 0.75])
for r in results:
    print(r.id, r.score)

# 4. Delete a point
crud.delete_points("cities", [2])
deleted = crud.delete_collection("cities2")
print(deleted)

