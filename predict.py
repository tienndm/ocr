from inference.realistic_predictor import RealisticPredictor

if __name__ == "__main__":
    # Example usage:
    predictor = RealisticPredictor(
        model_path="best_model.pth",
        vocab_path="data/vocab/vocab.json"
    )
    image_path = "data/N2/img/N2_3.bmp"
    caption = predictor.predict(image_path)
    print(caption)