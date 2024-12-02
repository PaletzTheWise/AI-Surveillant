import threading
import typing

_T = typing.TypeVar('T')

class Synchronized(typing.Generic[_T]):

    _value : _T
    _lock : threading.RLock

    def __init__(self, value : _T):
        self._value = value
        self._lock = threading.RLock()
    
    def lock(self) -> "LockContext[_T]":
        return LockContext(self)
    
    def set(self, value : _T) -> None:
        with self.lock():
            self._value = value

class LockContext(typing.Generic[_T]):

    _synchronized : Synchronized[_T]

    def __init__(self, _synchronized : Synchronized[_T] ):
        self._synchronized = _synchronized

    def __enter__(self):
        self._synchronized._lock.acquire()
        return self._synchronized._value

    def __exit__(self, type, value, traceback):
        return self._synchronized._lock.release()
