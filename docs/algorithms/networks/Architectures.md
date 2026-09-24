# Architectures

```python
from jaxnasium.algorithms import SAC
from jaxnasium.algorithms.architectures import BroNet

algorithm = SAC(critic_kwargs={"body": BroNet.with_params(depth=2, width_size=256)})
```

Any Equinox module following the same convention works too; these are just the bundled ones.

::: jaxnasium.algorithms.architectures.SimBa
    options:
        members:
            - __init__
            - with_params

::: jaxnasium.algorithms.architectures.BroNet
    options:
        members:
            - __init__
            - with_params

::: jaxnasium.algorithms.architectures.MLP
    options:
        members:
            - __init__
            - with_params

::: jaxnasium.algorithms.architectures.CNN
    options:
        members:
            - __init__
            - with_params

::: jaxnasium.algorithms.architectures.Identity
    options:
        members: false
