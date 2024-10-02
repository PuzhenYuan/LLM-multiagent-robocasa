try:
    raise ValueError('This is a value error')
except ValueError as e:
    print(f'Error: {e}')

print('stop')