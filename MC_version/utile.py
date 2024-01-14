import time



def execution_time(func):
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
    # # Using the decorator with the list as an argument
    # @execution_time(temps_list)
    # def my_function():
    #     # Your code here
    #     for _ in range(1000000):
    #         pass
    #
    #
    # # Calling the decorated function
    # my_function()
    # my_function()
    #
    # # Displaying the list of execution times
    # print("List of execution times:", temps_list)
