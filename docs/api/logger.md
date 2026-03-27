# Logger API

Logging is optional in the configuration model, but the logger family is threaded through the runtime when present. The logger is stored in `RuntimeContext`, exposed through `ALRuntimeMixin.logger`, used by the active-learning loop for round-level metrics in budget-constrained discovery, and sometimes used by components such as `BraninOracle` for figures.

## Modules at a glance

| Module | Main symbols | Role |
| --- | --- | --- |
| `activelearning.logger.logger` | `Logger`, `ConsoleLogger`, `WandbLogger`, `CometLogger`, `AimLogger`, `MultiLogger` | Runtime logging backends. |
| `activelearning.logger.config` | logger config models, `build_logger`, `bootstrap_logger_backend_imports` | YAML-facing builders and backend bootstrap helpers. |

## Core abstraction: `Logger`

Defined in `activelearning.logger.logger`.

| Method | Purpose |
| --- | --- |
| `log_config(config)` | Record the resolved experiment configuration. |
| `log_metric(key, value)` | Record a scalar or metric-like value. |
| `log_figure(key, figure)` | Record a figure or plot object. |
| `log_step(step)` | Flush or commit buffered data for a training or active-learning step. |
| `end()` | Finalize the logging session. |

If `run_name` is omitted, the base class defaults to a timestamp string.

## Concrete backends

| Class | Backend | Buffering model | Special behavior |
| --- | --- | --- | --- |
| `ConsoleLogger` | stdout | Buffers metrics until `log_step()` | Prints config as JSON and acknowledges figures without rendering them. |
| `WandbLogger` | Weights and Biases | Buffers metrics and figures until `log_step()` | Wraps figures as `wandb.Image`. |
| `CometLogger` | Comet ML | Logs metrics immediately, with step tracked separately | Non-numeric metric values are sent through `log_text(...)` instead of metric APIs. |
| `AimLogger` | Aim | Buffers until `log_step()` | Non-numeric values become `aim.Text`; matplotlib figures become `aim.Image`. |
| `MultiLogger` | fan-out wrapper | Delegates each call to child loggers | Has no project or run state of its own. |

## Configuration

Defined in `activelearning.logger.config`.

| Config model | Builds | Key fields |
| --- | --- | --- |
| `ConsoleLoggerConfig` | `ConsoleLogger(...)` | `project_name`, optional `run_name` |
| `WandbLoggerConfig` | `WandbLogger(...)` | `project_name`, optional `run_name` |
| `CometLoggerConfig` | `CometLogger(...)` | `project_name`, optional `run_name`, `workspace`, `api_key` |
| `AimLoggerConfig` | `AimLogger(...)` | `project_name`, optional `run_name`, `repo` |
| `MultiLoggerConfig` | `MultiLogger(loggers=[...])` | `loggers` |
| `LoggerConfig` | discriminated union | Includes the five rows above |

One configuration detail to note: `MultiLoggerConfig.loggers` accepts child configs from the four single-backend config classes only. The recursive child union does **not** include another `MultiLoggerConfig`.

## Helper functions

| Symbol | Purpose |
| --- | --- |
| `build_logger(config)` | Convenience wrapper that returns `None` when the config is `None`, otherwise `config.build()`. |
| `bootstrap_logger_backend_imports(raw_cfg)` | Scans the raw config tree and eagerly imports `comet_ml` when a `CometLogger` is referenced. |

The bootstrap helper exists because Comet warns about late imports after runtime-heavy modules such as torch have already been imported.

## Runtime integration

- `RuntimeConfig.build_context(logger=logger)` stores the logger in `RuntimeContext`.
- Every runtime-aware component can then access it through `self.logger`.
- `activelearning.active_learning.active_learning` currently logs `round`, `num_new_samples`, `best_candidate`, `round_cost`, `total_cost`, and `budget_remaining`.
- `BraninOracle` uses the same logger to emit a landscape figure after each query.

If you only need to know whether logging is present, check `logger is None` at the config layer or `self.logger is None` inside runtime-aware components.

## Class Reference

::: activelearning.logger.logger.Logger
    options:
      show_source: false
      heading_level: 3

::: activelearning.logger.logger.ConsoleLogger
    options:
      show_source: false
      heading_level: 3

::: activelearning.logger.logger.WandbLogger
    options:
      show_source: false
      heading_level: 3

::: activelearning.logger.logger.CometLogger
    options:
      show_source: false
      heading_level: 3

::: activelearning.logger.logger.AimLogger
    options:
      show_source: false
      heading_level: 3

::: activelearning.logger.logger.MultiLogger
    options:
      show_source: false
      heading_level: 3
