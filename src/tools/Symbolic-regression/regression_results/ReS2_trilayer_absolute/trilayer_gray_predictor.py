import numpy as np


def predict_trilayer_gray(layer1_gray, layer2_gray, layer3_gray):
    """
    Predict trilayer gray value from three monolayer gray values.

    Formula source
    --------------
    PySR symbolic regression.

    Formula complexity
    ------------------
    Node count: 7
    Tree depth: 0

    Input
    -----
    layer1_gray, layer2_gray, layer3_gray:
        Scalar, list, or NumPy array.

    Output
    ------
    Predicted trilayer gray value clipped to [0, 255].
    """
    layer1_gray = np.asarray(layer1_gray)
    layer2_gray = np.asarray(layer2_gray)
    layer3_gray = np.asarray(layer3_gray)

    trilayer_gray = (layer3_gray + (layer1_gray + layer2_gray)) * 0.5477291
    trilayer_gray = np.clip(trilayer_gray, 0, 255)

    if trilayer_gray.ndim == 0:
        return float(trilayer_gray)

    return trilayer_gray


if __name__ == "__main__":
    print("Single-sample test")
    pred = predict_trilayer_gray(128, 135, 140)
    print(f"layer1=128, layer2=135, layer3=140 -> trilayer={pred:.2f}")

    print("\nBatch test")
    l1_arr = np.array([100, 120, 140, 160])
    l2_arr = np.array([110, 130, 150, 170])
    l3_arr = np.array([105, 125, 145, 165])

    preds = predict_trilayer_gray(l1_arr, l2_arr, l3_arr)

    print(f"layer1: {l1_arr}")
    print(f"layer2: {l2_arr}")
    print(f"layer3: {l3_arr}")
    print(f"prediction: {preds.round(2)}")
