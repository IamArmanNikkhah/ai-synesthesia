# decoder.py

from common import spherical_harmonics_matrix
from position import Position
import numpy as np

DECODING_METHODS = ['projection', 'pseudoinv']
DEFAULT_DECODING = 'projection'

class AmbiDecoder:
    def __init__(self, speakers_pos, ambi_format, method=DEFAULT_DECODING):
        assert method in DECODING_METHODS
        if isinstance(speakers_pos, Position):
            speakers_pos = [speakers_pos]
        assert isinstance(speakers_pos, list) and all(isinstance(p, Position) for p in speakers_pos)
        self.speakers_pos = speakers_pos
        self.method = method

        # Use attributes from ambi_format
        self.order = ambi_format.order
        self.ordering = ambi_format.ordering
        self.normalization = ambi_format.normalization

        # Compute the spherical harmonics matrix
        self.sph_mat = spherical_harmonics_matrix(
            self.speakers_pos,
            self.order,
            self.ordering,
            self.normalization
        )

        if self.method == 'pseudoinv':
            self.pinv = np.linalg.pinv(self.sph_mat)

    def decode(self, ambi):
        if self.method == 'projection':
            return np.dot(ambi, self.sph_mat.T)
        elif self.method == 'pseudoinv':
            return np.dot(ambi, self.pinv)
        else:
            raise ValueError(f"Unknown decoding method: {self.method}")
