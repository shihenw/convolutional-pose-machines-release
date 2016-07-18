from config_reader import config_reader
from applymodel import applymodel
from visualize_save import visualize_save

if __name__ == "__main__":
    param, model = config_reader()
    test_image = '../sample_image/im1429.jpg'
    heatmaps, prediction = applymodel(test_image, param, model)
    visualize_save(test_image, heatmaps, prediction, param, model) # save images