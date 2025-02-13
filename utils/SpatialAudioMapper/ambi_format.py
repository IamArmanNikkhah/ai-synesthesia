# ambi_format.py
class AmbiFormat:
    def __init__(self, order, ordering='ACN', normalization='SN3D'):
        self.order = order
        self.ordering = ordering  # Channel ordering ('ACN' or 'FuMa')
        self.normalization = normalization  # Normalization scheme ('SN3D' or 'N3D')
        self.num_channels = (order + 1) ** 2
