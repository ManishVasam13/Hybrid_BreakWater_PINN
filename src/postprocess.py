import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

loss = pd.read_csv("results/loss_history.csv")
plt.figure()
plt.semilogy(loss["epoch"], loss["total"], label="total")
plt.semilogy(loss["epoch"], loss["pde"], label="PDE")
plt.semilogy(loss["epoch"], loss["free_surface"], label="free surface")
plt.semilogy(loss["epoch"], loss["bed"], label="bed")
plt.semilogy(loss["epoch"], loss["inlet"], label="inlet")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid(True)
plt.savefig("results/losses.png", dpi=200, bbox_inches="tight")
plt.close()

d = np.load("results/pinn_field.npz")
plt.figure(figsize=(10,4))
plt.imshow(d["phi"], extent=[d["x"].min(),d["x"].max(),d["z"].min(),d["z"].max()],
           origin="lower", aspect="auto")
plt.colorbar(label="phi")
plt.xlabel("x [m]")
plt.ylabel("z [m]")
plt.title("PINN field")
plt.savefig("results/field.png", dpi=200, bbox_inches="tight")
plt.close()

print("Postprocessing complete.")
print("Main outputs: results/losses.png, results/field.png, results/pinn_model.pt")
