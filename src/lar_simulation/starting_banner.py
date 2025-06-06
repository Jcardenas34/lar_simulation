
def simulation_start_end_announcement(func):
    '''
    A wrapper function that will decorate the starting and ending of the simulation
    '''
    def banner(*args, **kwargs):
        '''
        Places banner over and below function
        '''
        print("=== Now starting simulation ===\n")
        decorated_func = func(*args, **kwargs)
        print("=== End of simulation ===\n")
        return decorated_func


    return banner
