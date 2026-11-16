def parse_parameters(remainder):
    """Parse unknown CLI arguments of the form `--KEY1 VALUE1 --KEY2 VALUE2 ...` into a dict."""
    error = False
    parameters = {}
    if len(remainder) % 2 != 0:
        error = True
    else:
        for i in range(0, len(remainder), 2):
            if not remainder[i].startswith("--"):
                error = True
                break
            else:
                parameters[remainder[i][2:]] = remainder[i + 1]

    if error:
        raise RuntimeError("Invalid parameters specification! Must be of the form: --KEY1 VALUE --KEY2 VALUE2 ...")

    return parameters
