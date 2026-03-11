def calculate_symmetry(left_value, right_value):
    if left_value == 0 or right_value == 0:
        return 0

    symmetry = 1 - abs(left_value - right_value) / max(left_value, right_value)
    return round(symmetry * 100, 2)