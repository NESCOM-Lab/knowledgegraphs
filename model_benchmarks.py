import matplotlib.pyplot as plt
import numpy as np


x = np.array(["gpt-4.1-nano", "Ollama-local", "Gemini-2.0-flash-lite_b3", "Gemini-2.0-flash-lite_b25"])
y = np.array([370.3, 310.57, 103.59, 46.20, ])


plt.bar(x, y)
plt.xlabel("Models")
plt.ylabel("Time (s)")
plt.show()


