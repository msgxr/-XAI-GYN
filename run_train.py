from src.train import train, DEFAULT_CONFIG

if __name__ == "__main__":
    cfg = DEFAULT_CONFIG.copy()
    cfg['data_dir'] = 'data/processed'
    cfg['epochs'] = 5
    cfg['batch_size'] = 16
    cfg['val_split'] = 0.15
    cfg['test_split'] = 0.0
    train(cfg)
