import preprocess
import train
import embed
import autoencoder as ae
import tabulate
import argparse

class PipelineMain():
    def __init__(self, args):
        self.args = args

    def run(self):
        # Load and reprocess data
        FRAME_SIZE = 512
        HOP_LENGTH = 2048
        #DURATION = 0.74  # in seconds
        #SAMPLE_RATE = 22050
        # TODO: find correct values for these
        DURATION = 5
        SAMPLE_RATE = 16000
        MONO = True

        LEARNING_RATE = 0.0005
        BATCH_SIZE = 64
        EPOCHS = 15

        SPECTROGRAMS_PATH = "autoencoder/fsdd/spectograms"

        SPECTROGRAMS_SAVE_DIR = "autoencoder/fsdd/spectograms"
        MIN_MAX_VALUES_SAVE_DIR = "autoencoder/fsdd"
        FILES_DIR = "data/ECS50"
        # instantiate all objects
        loader = preprocess.Loader(SAMPLE_RATE, DURATION, MONO)
        padder = preprocess.Padder()
        log_spectrogram_extractor = preprocess.LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
        min_max_normaliser = preprocess.MinMaxNormaliser(0, 1)
        saver = preprocess.Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

        preprocessing_pipeline = preprocess.PreprocessingPipeline()
        preprocessing_pipeline.loader = loader
        preprocessing_pipeline.padder = padder
        preprocessing_pipeline.extractor = log_spectrogram_extractor
        preprocessing_pipeline.normaliser = min_max_normaliser
        preprocessing_pipeline.saver = saver

        preprocessing_pipeline.process(FILES_DIR)
        
        # Train model
        x_train, _ = train.load_fsdd(SPECTROGRAMS_PATH)
        autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
        autoencoder.save("autoencoder/runs/run3")
        
        # Load model
        vae = ae.VAE.load("autoencoder/runs/run2")
        
        # Generate embeddings
        embeddings, labels, file_paths = embed(vae, SPECTROGRAMS_PATH)
        
        # Evaluate embeddings
        euc_acc, cos_acc = embed.comps(embeddings, labels, file_paths)
        print("Accuracy Table for pretrained VAE model:\n")
        print(tabulate([['Euclidean', euc_acc], ['Cosine', cos_acc]], headers=['Similarity Type', 'Accuracy']))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    main = PipelineMain(args)
    main.run()