import time



def execution_time(func):
    """
    Decorator to measure the execution time of a function
    if the function is called multiple times, the execution time is added to a list in the instance
    :param func: function to measure the execution time
    :return: the decorated function
    """
    def wrapper(instance, *args, **kwargs):
        if not hasattr(instance, 'execution_times'):
            instance.execution_times = []  # Create the list if it doesn't exist in the instance yet
        start_time = time.time()
        result = func(instance, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        instance.execution_times.append(execution_time)
        return result

    return wrapper


if __name__ == '__main__':
    pass
