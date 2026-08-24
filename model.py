"""
JEPA World Model from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_env_state
import torch

def init_env_state(room_size: int = 8, seed: int | None = None) -> torch.Tensor:
    # Use a local generator so seeding does not affect global PyTorch RNG state.
    generator = torch.Generator()

    if seed is not None:
        generator.manual_seed(seed)

    # Sample two integer coordinates in [0, room_size - 1].
    state = torch.randint(low=0, high=room_size, size=(2,), generator=generator, dtype=torch.int64)

    # Convert to the required float32 state representation.
    return state.to(torch.float32)

# Step 2 - apply_action
def apply_action(state: torch.Tensor, action: int, room_size: int = 8) -> torch.Tensor:
    next_state = state.clone()

    if action == 0:
        next_state[1] -= 1
    elif action == 1:
        next_state[1] += 1
    elif action == 2:
        next_state[0] -= 1
    elif action == 3:
        next_state[0] += 1

    next_state = torch.clamp(next_state, 0, room_size - 1)

    return next_state

# Step 3 - render_observation
def render_observation(state: torch.Tensor, room_size: int = 8) -> torch.Tensor:
    observation = torch.zeros((1, room_size, room_size), dtype=torch.float32)

    x = int(state[0].item())
    y = int(state[1].item())

    observation[0, y, x] = 1.0

    return observation

# Step 4 - env_reset
def env_reset(room_size: int = 8, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    state = init_env_state(room_size=room_size, seed=seed)
    observation = render_observation(state=state, room_size=room_size)

    return state, observation

# Step 5 - env_step
def env_step(state: torch.Tensor, action: int, room_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    next_state = apply_action(state=state, action=action, room_size=room_size)
    next_observation = render_observation(state=next_state, room_size=room_size)

    return next_state, next_observation

# Step 6 - collect_random_transitions
def collect_random_transitions(num_transitions: int, room_size: int = 8, seed: int = 0) -> dict:
    if num_transitions == 0:
        return {
            "observations": torch.empty((0, 1, room_size, room_size), dtype=torch.float32),
            "actions": torch.empty((0,), dtype=torch.long),
            "next_observations": torch.empty((0, 1, room_size, room_size), dtype=torch.float32),
            "states": torch.empty((0, 2), dtype=torch.float32),
            "next_states": torch.empty((0, 2), dtype=torch.float32),
        }

    generator = torch.Generator().manual_seed(seed)

    state, observation = env_reset(room_size=room_size, seed=seed)

    observations = []
    actions = []
    next_observations = []
    states = []
    next_states = []

    for _ in range(num_transitions):
        action = torch.randint(low=0, high=4, size=(), generator=generator).item()

        next_state, next_observation = env_step(state=state, action=action, room_size=room_size)

        observations.append(observation)
        actions.append(action)
        next_observations.append(next_observation)
        states.append(state)
        next_states.append(next_state)

        state = next_state
        observation = next_observation

    return {
        "observations": torch.stack(observations),
        "actions": torch.tensor(actions, dtype=torch.long),
        "next_observations": torch.stack(next_observations),
        "states": torch.stack(states),
        "next_states": torch.stack(next_states),
    }

# Step 7 - build_transition_dataset
def build_transition_dataset(num_transitions: int = 512, room_size: int = 8, seed: int = 0) -> dict:
    return collect_random_transitions(num_transitions=num_transitions, room_size=room_size, seed=seed)

# Step 8 - init_encoder_params
def init_encoder_params(obs_channels: int = 1, room_size: int = 8, latent_dim: int = 32, seed: int = 0) -> dict:
    torch.manual_seed(seed)

    conv1_w = torch.randn(16, obs_channels, 3, 3) * 0.1
    conv1_b = torch.zeros(16)

    conv2_w = torch.randn(32, 16, 3, 3) * 0.1
    conv2_b = torch.zeros(32)

    spatial_size = room_size // 2
    fc_input_dim = 32 * spatial_size * spatial_size

    fc_w = torch.randn(latent_dim, fc_input_dim) * 0.1
    fc_b = torch.zeros(latent_dim)

    params = {
        "conv1_w": conv1_w,
        "conv1_b": conv1_b,
        "conv2_w": conv2_w,
        "conv2_b": conv2_b,
        "fc_w": fc_w,
        "fc_b": fc_b,
    }

    for param in params.values():
        param.requires_grad_(True)

    return params

# Step 9 - encoder_forward
import torch.nn.functional as F

def encoder_forward(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    x = F.conv2d(obs, encoder_params["conv1_w"], encoder_params["conv1_b"], stride=1, padding=1)
    x = F.relu(x)

    x = F.conv2d(x, encoder_params["conv2_w"], encoder_params["conv2_b"], stride=2, padding=1)
    x = F.relu(x)

    x = x.flatten(start_dim=1)

    x = F.linear(x, encoder_params["fc_w"], encoder_params["fc_b"])

    return x

# Step 10 - init_target_encoder
def init_target_encoder(encoder_params: dict) -> dict:
    return {key: value.detach().clone() for key, value in encoder_params.items()}

# Step 11 - ema_update
def ema_update(target_params: dict, encoder_params: dict, tau: float = 0.99) -> dict:
    return {key: tau * target_params[key] + (1.0 - tau) * encoder_params[key] for key in target_params}

# Step 12 - encode_batch
def encode_batch(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    return encoder_forward(obs, encoder_params)

# Step 13 - init_predictor_params
def init_predictor_params(latent_dim: int = 32, action_dim: int = 4, hidden_dim: int = 64, seed: int = 0) -> dict:
    torch.manual_seed(seed)

    params = {
        "action_embed_w": torch.randn(action_dim, latent_dim) * 0.02,
        "fc1_w": torch.randn(hidden_dim, 2 * latent_dim) * 0.02,
        "fc1_b": torch.zeros(hidden_dim),
        "fc2_w": torch.randn(latent_dim, hidden_dim) * 0.02,
        "fc2_b": torch.zeros(latent_dim),
    }

    for param in params.values():
        param.requires_grad_(True)

    return params

# Step 14 - embed_action
def embed_action(actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    return predictor_params["action_embed_w"][actions]

# Step 15 - predictor_forward
def predictor_forward(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    action_embeddings = embed_action(actions, predictor_params)
    x = torch.cat([embeddings, action_embeddings], dim=1)
    x = torch.relu(x @ predictor_params["fc1_w"].T + predictor_params["fc1_b"])
    x = x @ predictor_params["fc2_w"].T + predictor_params["fc2_b"]

    return x

# Step 16 - predict_next_embedding
def predict_next_embedding(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    return predictor_forward(embeddings, actions, predictor_params)

# Step 17 - prediction_loss
def prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((predicted - target) ** 2)

# Step 18 - variance_loss
def variance_loss(embeddings: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(torch.var(embeddings, dim=0, unbiased=True) + eps)
    return torch.mean(torch.relu(gamma - std))

# Step 19 - covariance_loss
def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (embeddings.shape[0] - 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))

    return torch.sum(off_diagonal ** 2) / embeddings.shape[1]

# Step 20 - vicreg_regularizer
def vicreg_regularizer(embeddings: torch.Tensor, var_weight: float = 1.0, cov_weight: float = 0.04, gamma: float = 1.0) -> torch.Tensor:
    return var_weight * variance_loss(embeddings, gamma=gamma) + cov_weight * covariance_loss(embeddings)

# Step 21 - jepa_loss
def jepa_loss(predicted: torch.Tensor, target: torch.Tensor, online_embeddings: torch.Tensor, pred_weight: float = 1.0, var_weight: float = 1.0, cov_weight: float = 0.04) -> torch.Tensor:
    prediction_term = prediction_loss(predicted, target)
    regularization_term = vicreg_regularizer(online_embeddings, var_weight=var_weight, cov_weight=cov_weight)

    return pred_weight * prediction_term + regularization_term

# Step 22 - collapse_metric
def collapse_metric(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.std(embeddings, dim=0).mean()

# Step 23 - jepa_training_step
def jepa_training_step(batch: dict, encoder_params: dict, target_params: dict, predictor_params: dict, lr: float = 1e-3, tau: float = 0.99) -> tuple[dict, dict, dict, float, float]:
    observations = batch["observations"]
    actions = batch["actions"]
    next_observations = batch["next_observations"]

    online_embeddings = encoder_forward(observations, encoder_params)

    with torch.no_grad():
        target_embeddings = encoder_forward(next_observations, target_params)

    predicted_embeddings = predictor_forward(online_embeddings, actions, predictor_params)

    loss = jepa_loss(predicted_embeddings, target_embeddings, online_embeddings)

    encoder_keys = list(encoder_params.keys())
    predictor_keys = list(predictor_params.keys())
    encoder_values = [encoder_params[key] for key in encoder_keys]
    predictor_values = [predictor_params[key] for key in predictor_keys]

    gradients = torch.autograd.grad(loss, encoder_values + predictor_values)

    encoder_grads = gradients[:len(encoder_values)]
    predictor_grads = gradients[len(encoder_values):]

    updated_encoder_params = {
        key: (param - lr * grad).detach().clone().requires_grad_(True)
        for key, param, grad in zip(encoder_keys, encoder_values, encoder_grads)
    }

    updated_predictor_params = {
        key: (param - lr * grad).detach().clone().requires_grad_(True)
        for key, param, grad in zip(predictor_keys, predictor_values, predictor_grads)
    }

    updated_target_params = ema_update(target_params, updated_encoder_params, tau=tau)

    loss_value = float(loss.detach().item())
    collapse_value = float(collapse_metric(online_embeddings.detach()).item())

    return updated_encoder_params, updated_target_params, updated_predictor_params, loss_value, collapse_value

# Step 24 - train_jepa
def train_jepa(dataset: dict, encoder_params: dict, target_params: dict, predictor_params: dict, num_steps: int = 50, batch_size: int = 32, lr: float = 1e-3, tau: float = 0.99, seed: int = 0) -> tuple[dict, dict, dict, list]:
    generator = torch.Generator().manual_seed(seed)
    num_samples = dataset["observations"].shape[0]

    history = []

    for _ in range(num_steps):
        indices = torch.randint(0, num_samples, (batch_size,), generator=generator)

        batch = {
            "observations": dataset["observations"][indices],
            "actions": dataset["actions"][indices],
            "next_observations": dataset["next_observations"][indices],
        }

        encoder_params, target_params, predictor_params, loss, collapse = jepa_training_step(
            batch,
            encoder_params,
            target_params,
            predictor_params,
            lr=lr,
            tau=tau,
        )

        history.append({
            "loss": loss,
            "collapse": collapse,
        })

    return encoder_params, target_params, predictor_params, history

# Step 25 - rollout_latent_dynamics
def rollout_latent_dynamics(initial_embedding: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    batched = initial_embedding.dim() == 2

    if batched:
        batch_size = initial_embedding.shape[0]

        if actions.dim() == 1:
            actions = actions.unsqueeze(0).expand(batch_size, -1)
    else:
        if actions.dim() != 1:
            actions = actions.squeeze(0)

        initial_embedding = initial_embedding.unsqueeze(0)
        actions = actions.unsqueeze(0)

    trajectory = [initial_embedding]
    current_embedding = initial_embedding

    for t in range(actions.shape[1]):
        current_embedding = predict_next_embedding(current_embedding, actions[:, t], predictor_params)
        trajectory.append(current_embedding)

    trajectory = torch.stack(trajectory, dim=0)

    if not batched:
        trajectory = trajectory[:, 0, :]

    return trajectory

# Step 26 - multi_step_prediction_error
def multi_step_prediction_error(dataset: dict, encoder_params: dict, target_params: dict, predictor_params: dict, horizon: int = 5, num_samples: int = 32) -> float:
    num_transitions = dataset["observations"].shape[0]
    num_starts = min(num_samples, num_transitions - horizon)

    if num_starts <= 0 or horizon <= 0:
        return 0.0

    errors = []

    with torch.no_grad():
        for start in range(num_starts):
            start_observation = dataset["observations"][start].unsqueeze(0)
            actions = dataset["actions"][start:start + horizon].unsqueeze(0)
            future_observations = dataset["next_observations"][start:start + horizon]

            initial_embedding = encoder_forward(start_observation, encoder_params)
            predicted_trajectory = rollout_latent_dynamics(initial_embedding, actions, predictor_params)

            target_embeddings = encoder_forward(future_observations, target_params)
            predicted_embeddings = predicted_trajectory[1:]

            mse = torch.mean((predicted_embeddings - target_embeddings) ** 2)
            errors.append(mse)

    return float(torch.stack(errors).mean().item())

# Step 27 - init_linear_probe
def init_linear_probe(latent_dim: int = 32, state_dim: int = 2, seed: int = 0) -> dict:
    torch.manual_seed(seed)

    probe = {
        "w": torch.randn(state_dim, latent_dim) * 0.01,
        "b": torch.zeros(state_dim),
    }

    return probe

# Step 28 - train_linear_probe
def train_linear_probe(embeddings: torch.Tensor, states: torch.Tensor, probe_params: dict, num_steps: int = 100, lr: float = 1e-2) -> dict:
    w = probe_params["w"].detach().clone()
    b = probe_params["b"].detach().clone()

    for _ in range(num_steps):
        predictions = embeddings @ w + b
        error = predictions - states

        grad_w = 2.0 * (embeddings.T @ error) / error.numel()
        grad_b = 2.0 * error.sum(dim=0) / error.numel()

        w = w - lr * grad_w
        b = b - lr * grad_b

    return {
        "w": w,
        "b": b,
    }

# Step 29 - probe_state_recovery
def probe_state_recovery(dataset: dict, encoder_params: dict, probe_params: dict | None = None, num_probe_steps: int = 100) -> dict:
    observations = dataset["observations"]
    states = dataset["states"].float()

    with torch.no_grad():
        embeddings = encode_batch(observations, encoder_params)

    if probe_params is None:
        probe_params = init_linear_probe(latent_dim=embeddings.shape[1], state_dim=states.shape[1], seed=0)

    train_params = {
        "w": probe_params["w"].T.detach().clone(),
        "b": probe_params["b"].detach().clone(),
    }

    train_params = train_linear_probe(
        embeddings,
        states,
        train_params,
        num_steps=num_probe_steps,
    )

    trained_probe_params = {
        "w": train_params["w"].T,
        "b": train_params["b"],
    }

    with torch.no_grad():
        predictions = embeddings @ trained_probe_params["w"].T + trained_probe_params["b"]
        errors = predictions - states
        mse = torch.mean(errors ** 2)
        mean_abs_error = torch.mean(torch.abs(errors))

    return {
        "mse": float(mse.item()),
        "mean_abs_error": float(mean_abs_error.item()),
        "probe_params": trained_probe_params,
    }

# Step 30 - encode_goal
def encode_goal(goal_state: torch.Tensor, encoder_params: dict, room_size: int = 8) -> torch.Tensor:
    observation = render_observation(goal_state, room_size=room_size)
    embedding = encode_batch(observation.unsqueeze(0), encoder_params)
    return embedding.squeeze(0)

# Step 31 - latent_cost
def latent_cost(latents, goal_embedding):
    return torch.sum((latents - goal_embedding) ** 2, dim=-1)

# Step 32 - sample_action_sequences
def sample_action_sequences(n_sequences, horizon, n_actions):
    return torch.randint(low=0, high=n_actions, size=(n_sequences, horizon), dtype=torch.long)

# Step 33 - score_action_sequences
def score_action_sequences(start_embedding, action_sequences, goal_embedding, predictor_params):
    n_sequences, horizon = action_sequences.shape

    embeddings = start_embedding.unsqueeze(0).expand(n_sequences, -1)
    total_cost = torch.zeros(n_sequences, dtype=embeddings.dtype, device=embeddings.device)

    for t in range(horizon):
        embeddings = predict_next_embedding(embeddings, action_sequences[:, t], predictor_params)
        total_cost = total_cost + latent_cost(embeddings, goal_embedding)

    return total_cost

# Step 34 - select_best_plan
def select_best_plan(action_sequences, costs):
    best_index = torch.argmin(costs)
    return action_sequences[best_index]

# Step 35 - mpc_step
def mpc_step(start_embedding, goal_embedding, predictor_params, n_sequences, horizon, n_actions):
    action_sequences = sample_action_sequences(n_sequences, horizon, n_actions)
    costs = score_action_sequences(start_embedding, action_sequences, goal_embedding, predictor_params)
    best_plan = select_best_plan(action_sequences, costs)

    return int(best_plan[0].item())

# Step 36 - run_mpc_episode
def run_mpc_episode(encoder_params, predictor_params, goal_pos, room_size, agent_size, max_steps, n_sequences, horizon, n_actions):
    if isinstance(goal_pos, torch.Tensor):
        goal_state = goal_pos.to(dtype=torch.float32)
    else:
        goal_state = torch.tensor(goal_pos, dtype=torch.float32)

    state, _ = env_reset(room_size=room_size)
    trajectory = [tuple(state.tolist())]

    goal_embedding = encode_goal(goal_state, encoder_params, room_size=room_size)

    for _ in range(max_steps):
        if torch.equal(state, goal_state):
            break

        observation = render_observation(state, room_size=room_size)
        start_embedding = encode_batch(observation.unsqueeze(0), encoder_params).squeeze(0)

        action = mpc_step(
            start_embedding,
            goal_embedding,
            predictor_params,
            n_sequences,
            horizon,
            n_actions,
        )

        state, _ = env_step(state, action, room_size=room_size)
        trajectory.append(tuple(state.tolist()))

    final_distance = torch.norm(state - goal_state).item()
    success = bool(torch.equal(state, goal_state))

    return {
        "success": success,
        "steps": len(trajectory) - 1,
        "trajectory": trajectory,
        "final_distance": float(final_distance),
    }

# Step 37 - evaluate_planner
def evaluate_planner(encoder_params, predictor_params, room_size, agent_size, n_episodes, max_steps, n_sequences, horizon, n_actions):
    successes = []
    steps = []
    final_distances = []

    for _ in range(n_episodes):
        goal_pos = torch.randint(0, room_size, (2,)).float()

        result = run_mpc_episode(encoder_params, predictor_params, goal_pos, room_size, agent_size, max_steps, n_sequences, horizon, n_actions)

        successes.append(float(result["success"]))
        steps.append(result["steps"])
        final_distances.append(result["final_distance"])

    return {
        "success_rate": sum(successes) / n_episodes if n_episodes > 0 else 0.0,
        "mean_steps": sum(steps) / n_episodes if n_episodes > 0 else 0.0,
        "mean_final_distance": sum(final_distances) / n_episodes if n_episodes > 0 else 0.0,
    }

# Step 38 - jepa_world_model_experiment
def jepa_world_model_experiment(room_size, agent_size, embed_dim, n_train_transitions, n_epochs, batch_size, n_probe_samples, n_eval_episodes, max_steps, n_sequences, horizon):
    dataset = build_transition_dataset(
        num_transitions=n_train_transitions,
        room_size=room_size,
        seed=0,
    )

    encoder_params = init_encoder_params(
        obs_channels=1,
        room_size=room_size,
        latent_dim=embed_dim,
        seed=0,
    )

    target_params = init_target_encoder(encoder_params)

    predictor_params = init_predictor_params(
        latent_dim=embed_dim,
        action_dim=4,
        hidden_dim=64,
        seed=0,
    )

    encoder_params, target_params, predictor_params, history = train_jepa(
        dataset,
        encoder_params,
        target_params,
        predictor_params,
        num_steps=n_epochs,
        batch_size=batch_size,
        lr=1e-3,
        tau=0.99,
        seed=0,
    )

    probe_dataset = {
        "observations": dataset["observations"][:n_probe_samples],
        "states": dataset["states"][:n_probe_samples],
    }

    probe_result = probe_state_recovery(
        probe_dataset,
        encoder_params,
        num_probe_steps=100,
    )

    probe_mse = probe_result["mse"]
    probe_states = probe_dataset["states"].float()
    state_mean = torch.mean(probe_states, dim=0)
    state_variance = torch.mean((probe_states - state_mean) ** 2).item()

    probe_r2 = 1.0 - probe_mse / (state_variance + 1e-8)

    planner_result = evaluate_planner(
        encoder_params,
        predictor_params,
        room_size,
        agent_size,
        n_eval_episodes,
        max_steps,
        n_sequences,
        horizon,
        4,
    )

    return {
        "train_losses": [step["loss"] for step in history],
        "collapse_metrics": [step["collapse"] for step in history],
        "probe_r2": float(probe_r2),
        "success_rate": float(planner_result["success_rate"]),
        "mean_steps": float(planner_result["mean_steps"]),
    }

