import openpi.shared.download as download
import openpi.training.config as _config


def main(config: _config.TrainConfig):

    save_path = download.maybe_download(config.weight_loader.params_path)
    print(f"Saved pi weights to:\n{save_path}")

    # NOTE: Hard coded the most popular tokenizer. May need to be adjusted for alternative tokenizers
    save_path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    print(f"Saved tokenizer weights to:\n{save_path}")


if __name__ == "__main__":
    for config in _config._CONFIGS_DICT.values():
        config.exp_name = "dummy_exp"

    main(_config.cli())
