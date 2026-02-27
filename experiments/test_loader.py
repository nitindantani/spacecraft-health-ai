from core.build_loader import get_loader


print("Creating loader...")

loader = get_loader(batch_size=64)

print("Loader created ✔")

# take one batch
for batch in loader:
    print("Batch shape:", batch.shape)
    break

print("Test finished ✔")
