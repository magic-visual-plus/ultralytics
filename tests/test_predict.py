from ultralytics import YOLO


def test_predict():
    checkpoint_path = '/root/.cache/trains/a4d98757-4f07-407c-9333-926a17b0b630/output/weights/best.pt'
    model = YOLO(checkpoint_path)

    config_dict = {
                "save": False,
                "show_labels": False,
                "verbose": False,
                "conf": .25,
                "retina_masks": True,
            }
    source = '/tmp/a.png'
    result = model.predict(source=source, **config_dict)[0]
    print(result)
    
def test_ssim():
    import torch
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    preds = torch.rand([256, 256])
    target = preds * 5.75
    # metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    # value = metric(preds, target)
    # print(value)
    
    from skimage.metrics import structural_similarity as ssim

    cur_ssim = ssim(
        target.numpy(),
        preds.numpy(),
        win_size=11,
        data_range=1.0,
    )
    print(cur_ssim)
    
if __name__ == '__main__':
    test_ssim()
    # test_predict()