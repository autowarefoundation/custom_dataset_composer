import yaml
from src.dataset_integration.dataset_integrator import DatasetIntegrator

if __name__ == '__main__':
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    dataset_integrator = DatasetIntegrator(config)



