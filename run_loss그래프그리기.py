

with open(file='loss_values.pickle', mode='rb') as f:
    loss_values=pickle.load(f)
plt.figure(figsize=(10,5))
plt.plot(range(100, 100*len(loss_values)+1, 100), loss_values, label="Training loss every 100 steps")
plt.title("Loss Values at Every 100 Steps")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.show()

