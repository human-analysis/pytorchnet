# verification.py

__all__ = ['TarAtFar']


class TarAtFar:
    def __init__(self):
        pass

    def __call__(self, verification, far=0.1):
        num = verification.size(0)
        for i in range(num):
            if verification[i, 1] > far:
                return 100 - verification[i, 0]
        return 0.0
