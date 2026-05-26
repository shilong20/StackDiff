import numpy as np


def predict_bilayer_gray(layer1_gray, layer2_gray):
    """
    Predict bilayer gray value from layer1 and layer2 gray values.

    Formula source
    --------------
    PySR symbolic regression.

    Formula complexity
    ------------------
    Node count: 5
    Tree depth: 0

    Input
    -----
    layer1_gray, layer2_gray:
        Scalar, list, or NumPy array.

    Output
    ------
    Predicted bilayer gray value clipped to [0, 255].
    """
    layer1_gray = np.asarray(layer1_gray)
    layer2_gray = np.asarray(layer2_gray)

    bilayer_gray = (layer1_gray + layer2_gray) * 0.9673063
    bilayer_gray = np.clip(bilayer_gray, 0, 255)

    if bilayer_gray.ndim == 0:
        return float(bilayer_gray)

    return bilayer_gray


if __name__ == "__main__":
    print("Single-sample test")
    pred = predict_bilayer_gray(128, 135)
    print(f"layer1=128, layer2=135 -> bilayer={pred:.2f}")

    print("\nBatch test")
    l1_arr = np.array([100, 120, 140, 160])
    l2_arr = np.array([110, 130, 150, 170])
    preds = predict_bilayer_gray(l1_arr, l2_arr)

    print(f"layer1: {l1_arr}")
    print(f"layer2: {l2_arr}")
    print(f"prediction: {preds.round(2)}")
