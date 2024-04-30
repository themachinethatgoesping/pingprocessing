def assert_length(function_name, base_vector, compare_vectors):
    for vec in compare_vectors:
        len(vec) == len(base_vector), f"ERROR[{function_name}]: error size mismatch! {len(vec)} != {len(base_vector)}"


def assert_valid_argument(function_name, arg, valid_args):
    assert arg in valid_args, f"ERROR[{function_name}]: unknown value '{arg}', use any of '{valid_args}'"
