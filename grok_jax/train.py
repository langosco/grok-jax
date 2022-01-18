# Some of the code here is adapted from
# https://github.com/deepmind/dm-haiku/blob/main/examples/transformer/model.py 

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, value_and_grad, jit, random
from typing import Optional, Mapping, Any

import optax
import functools

import wandb

from tqdm import tqdm
import itertools

import hydra
from omegaconf import DictConfig, OmegaConf

from grok_jax.transformer import build_forward_fn
from grok_jax.data import ArithmeticDataset, ArithmeticIterator

VOCAB_SIZE = 120

def accuracy(forward: callable,
             params,
             data: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Compute the accuracy on data wrt params.
    Mask out all tokens except the last one (ie the RHS of equation).
    """
    logits = forward(params, None, data['text'], is_training=False)
    y_pred = jnp.argmax(logits, axis=-1)
    y_pred_rhs = y_pred[:, -2]  # prediction for RHS
    
    targets_rhs = data['target'][:, -2]
    return jnp.mean(targets_rhs == y_pred_rhs)


def lm_loss_fn(forward_fn,
               vocab_size: int,
               params,
               rng,
               data: Mapping[str, jnp.ndarray],
               is_training: bool = True) -> jnp.ndarray:
    """
    Compute the loss on data wrt params.
    Mask out all tokens except the last one (ie the RHS of equation).
    """
    logits = forward_fn(params, rng, data['text'], is_training)
    targets = jax.nn.one_hot(data['target'], vocab_size)
    assert logits.shape == targets.shape

    # mask = jnp.greater(data['text'], 0)
    # what counts is the prediction for the last token (ie second-to-last logit)
    batch_size = data['text'].shape[0]
    mask = np.array([[False, False, False, False, True, False]] * batch_size)

    loss = -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
    loss = jnp.sum(loss * mask) / jnp.sum(mask)

    return loss


class Updater:
    """A stateless abstraction around an init_fn/update_fn pair.
    This extracts some common boilerplate from the training loop.
    """
    def __init__(self, net_init, loss_fn, accuracy_fn, optimizer):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._accuracy_fn = accuracy_fn
        self._opt = optimizer

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, rng, data):
        """Initializes state of the updater."""
        out_rng, init_rng = jax.random.split(rng)
        params = self._net_init(init_rng, data['text'])
        opt_state = self._opt.init(params)
        out = dict(
            step=np.array(0),
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        )
        return out

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: Mapping[str, Any], data: Mapping[str, jnp.ndarray]):
        """Updates the state using some data and returns metrics."""
        rng, new_rng = jax.random.split(state['rng'])
        params = state['params']
        loss, g = jax.value_and_grad(self._loss_fn)(params, rng, data)

        updates, opt_state = self._opt.update(g, state['opt_state'], params)
        params = optax.apply_updates(params, updates)

        new_state = {
            'step': state['step'] + 1,
            'rng': new_rng,
            'opt_state': opt_state,
            'params': params,
        }

        metrics = {
            'step': state['step'],
            'train/loss': loss,
        }
        return new_state, metrics
    
    @functools.partial(jax.jit, static_argnums=0)
    def validate(self, state: Mapping[str, Any], val_data: Mapping[str, jnp.ndarray]):
        params = state['params']
        loss = self._loss_fn(params, None, val_data, is_training=False)
        
        val_metrics = {
            'step': state['step']-1,
            'validation/loss': loss,
            'validation/accuracy': self._accuracy_fn(params, val_data),
        }
        return val_metrics

def train(config: Mapping[str, Any]) -> None:
    data_config = config["data"]
    train_config = config["train"]
    if config["seed"] is None:
        raise ValueError("Need to set seed")
    else:
        rng_key = random.PRNGKey(config["seed"])


    # prepare data
    train_data, val_data = ArithmeticDataset.splits(operator="-", train_pct=data_config["train_percent"])
    train_data = ArithmeticIterator(train_data, device=config["device"])
    train_data = itertools.cycle([{k: v.numpy() for k, v in batch.items()} for batch in train_data])
    # TODO: .numpy() puts data on CPU I think, which we probably want to avoid

    val_data = ArithmeticIterator(val_data, device=config["device"], batchsize_hint=-1)
    val_data = next(val_data)  # only one element
    val_data = {k: v.numpy() for k, v in val_data.items()}


    # prepare model and optimizer
    forward_fn = build_forward_fn(**config["model"], vocab_size=VOCAB_SIZE)
    loss_fn = functools.partial(lm_loss_fn, forward_fn.apply, VOCAB_SIZE)
    accuracy_fn = jit(functools.partial(accuracy, forward_fn.apply))

    lr = train_config["learning_rate"]
    num_warmup_steps = train_config["warmup_steps"]
    warmup_schedule = optax.linear_schedule(
            init_value=lr/num_warmup_steps, 
            end_value=lr, 
            transition_steps=num_warmup_steps)

    optimizer = optax.chain(
	    optax.adamw(1., b1=0.9, b2=0.98, weight_decay=train_config["weight_decay"]),
            optax.scale_by_schedule(warmup_schedule))

    updater = Updater(forward_fn.init, loss_fn, accuracy_fn, optimizer)

    key, subkey = random.split(rng_key)
    data = next(train_data)
    state = updater.init(subkey, data)

    wandb_cfg = config["wandb"]
    wandb.init(project=wandb_cfg["project_name"], config=config, tags=wandb_cfg["tags"])
    for _ in tqdm(range(train_config["num_steps"]),
                        disable=not train_config["progress_bar"]):
        data = next(train_data)
        state, metrics = updater.update(state, data)

        if metrics['step'] % train_config["log_every"] == 0:
            val_metrics = updater.validate(state, val_data)
            assert val_metrics['step'] == metrics['step']
            metrics.update(val_metrics)
            metrics.update({'train/accuracy': accuracy_fn(state['params'], data)})
            wandb.log(metrics)
