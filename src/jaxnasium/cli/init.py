import warnings

import create_rl_app.cli


def main():
    # deprecation warning
    warnings.warn(
        "The Jaxnasium CLI is deprecated and will be removed in a future version. Please use `create-rl-app` instead. \n"
        + "This command currently calls `create-rl-app` internally.",
        DeprecationWarning,
        stacklevel=2,
    )
    create_rl_app.cli.main()


if __name__ == "__main__":
    main()
