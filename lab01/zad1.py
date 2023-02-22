def prime(n):
    if n == 1:
        return False
    if n == 2:
        return True
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

print(prime(49))

def select_primes(n):
    primes = [num for num in n if prime(num)]
    return primes


print(select_primes([3,6,11,25,19]))