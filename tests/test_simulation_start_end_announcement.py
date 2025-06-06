from lar_simulation.starting_banner import simulation_start_end_announcement


@simulation_start_end_announcement
def simple_function(phrase):
    print(f"Testing simple phrase: {phrase}")
    return phrase


simple_function("test")

