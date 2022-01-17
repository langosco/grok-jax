from grok_jax.train import train
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    cfg["train"]["max_steps"] = int(cfg["train"]["max_steps"])
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    print()
    train(cfg)

if __name__ == "__main__":
    main()
