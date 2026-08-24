# JEPA World Model from Scratch

Build an action-conditioned Joint Embedding Predictive Architecture (JEPA) world model in pure PyTorch, following LeCun's path to autonomous intelligence. An encoder and EMA target learn collapse-resistant latents of a 2D room via VICReg; a predictor rolls dynamics in embedding space; acting is random-shooting MPC toward goal embeddings—no decoder, no pixels, no learned policy.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** init_env_state
- [x] **2.** apply_action
- [x] **3.** render_observation
- [x] **4.** env_reset
- [x] **5.** env_step
- [x] **6.** collect_random_transitions
- [x] **7.** build_transition_dataset
- [x] **8.** init_encoder_params
- [x] **9.** encoder_forward
- [x] **10.** init_target_encoder
- [x] **11.** ema_update
- [x] **12.** encode_batch
- [x] **13.** init_predictor_params
- [x] **14.** embed_action
- [x] **15.** predictor_forward
- [x] **16.** predict_next_embedding
- [x] **17.** prediction_loss
- [x] **18.** variance_loss
- [x] **19.** covariance_loss
- [x] **20.** vicreg_regularizer
- [x] **21.** jepa_loss
- [x] **22.** collapse_metric
- [x] **23.** jepa_training_step
- [x] **24.** train_jepa
- [x] **25.** rollout_latent_dynamics
- [x] **26.** multi_step_prediction_error
- [x] **27.** init_linear_probe
- [x] **28.** train_linear_probe
- [x] **29.** probe_state_recovery
- [x] **30.** encode_goal
- [x] **31.** latent_cost
- [x] **32.** sample_action_sequences
- [x] **33.** score_action_sequences
- [x] **34.** select_best_plan
- [x] **35.** mpc_step
- [x] **36.** run_mpc_episode
- [x] **37.** evaluate_planner
- [x] **38.** jepa_world_model_experiment

## Results

```
transitions: 512 | obs shape: (512, 1, 6, 6)
collapse metric BEFORE training: 0.035  (near 0 = collapsed)
JEPA loss: 0.964 -> 0.814
collapse metric AFTER training:  1.086  (target std is 1.0)
5-step latent prediction MSE (trained): 1.180
  (an untrained, collapsed encoder scores ~0 here -- trivially easy and useless)
linear probe on agent position: mean abs error = 0.56 cells (room is 6x6)
MPC planner:   success 70%, mean final distance 1.14
random policy: success 40%
```
