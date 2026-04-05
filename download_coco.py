import urllib.request

# Train images
urllib.request.urlretrieve(
    "http://images.cocodataset.org/zips/train2017.zip",
    "data/train2017.zip"
)

# Val images
urllib.request.urlretrieve(
    "http://images.cocodataset.org/zips/val2017.zip",
    "data/val2017.zip"
)

# Annotations
urllib.request.urlretrieve(
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "data/annotations_trainval2017.zip"
)

print("✅ Download complete! Files saved in /data")