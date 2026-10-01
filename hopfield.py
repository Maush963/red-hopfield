"""Red de Hopfield implementada solo con listas, ciclos y condicionales."""


class HopfieldNetwork:
    """Red de Hopfield para patrones representados con -1 y 1."""

    def __init__(self, size):
        if size <= 0:
            raise ValueError("El tamano debe ser mayor que cero.")

        self.size = size
        self.weights = []
        row = 0
        while row < size:
            current_row = []
            column = 0
            while column < size:
                current_row.append(0)
                column += 1
            self.weights.append(current_row)
            row += 1

    def train(self, patterns):
        """Calcula los pesos con la regla de Hebb."""
        if len(patterns) == 0:
            raise ValueError("Debe existir al menos un patron.")

        pattern_index = 0
        while pattern_index < len(patterns):
            self._validate_pattern(patterns[pattern_index])
            pattern_index += 1

        row = 0
        while row < self.size:
            column = 0
            while column < self.size:
                if row == column:
                    self.weights[row][column] = 0
                else:
                    total = 0
                    pattern_index = 0
                    while pattern_index < len(patterns):
                        total += patterns[pattern_index][row] * patterns[pattern_index][column]
                        pattern_index += 1
                    self.weights[row][column] = total
                column += 1
            row += 1

    def recall(self, pattern, max_steps=20):
        """Recupera un patron y devuelve todos los estados visitados."""
        self._validate_pattern(pattern)
        if max_steps <= 0:
            raise ValueError("max_steps debe ser mayor que cero.")

        current = pattern[:]
        history = [current[:]]
        step = 0

        while step < max_steps:
            next_state = []
            row = 0
            while row < self.size:
                activation = 0
                column = 0
                while column < self.size:
                    activation += self.weights[row][column] * current[column]
                    column += 1

                if activation > 0:
                    next_state.append(1)
                elif activation < 0:
                    next_state.append(-1)
                else:
                    next_state.append(current[row])
                row += 1

            history.append(next_state[:])
            if next_state == current:
                return history

            current = next_state
            step += 1

        return history

    def energy(self, pattern):
        """Calcula E = -1/2 * patron * pesos * patron."""
        self._validate_pattern(pattern)
        total = 0
        row = 0
        while row < self.size:
            column = 0
            while column < self.size:
                total += self.weights[row][column] * pattern[row] * pattern[column]
                column += 1
            row += 1
        return -total / 2

    def _validate_pattern(self, pattern):
        if len(pattern) != self.size:
            raise ValueError("El patron debe tener el tamano de la red.")

        index = 0
        while index < len(pattern):
            if pattern[index] != -1 and pattern[index] != 1:
                raise ValueError("Cada valor del patron debe ser -1 o 1.")
            index += 1


def print_pattern(pattern, columns):
    """Muestra un patron plano como una cuadricula."""
    index = 0
    while index < len(pattern):
        if pattern[index] == 1:
            print("##", end="")
        else:
            print("  ", end="")

        if (index + 1) % columns == 0:
            print()
        index += 1


def main():
    # Dos letras simples de 5 x 5 para mostrar almacenamiento y recuperacion.
    letter_x = [
        1, -1, -1, -1, 1,
        -1, 1, -1, 1, -1,
        -1, -1, 1, -1, -1,
        -1, 1, -1, 1, -1,
        1, -1, -1, -1, 1,
    ]
    letter_o = [
        -1, 1, 1, 1, -1,
        1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        -1, 1, 1, 1, -1,
    ]
    damaged_x = [
        1, -1, -1, -1, 1,
        -1, -1, -1, 1, -1,
        -1, -1, 1, -1, -1,
        -1, 1, -1, -1, -1,
        1, -1, -1, -1, 1,
    ]

    network = HopfieldNetwork(25)
    network.train([letter_x, letter_o])
    history = network.recall(damaged_x)
    recovered = history[-1]

    print("Patron danado:")
    print_pattern(damaged_x, 5)
    print("\nPatron recuperado:")
    print_pattern(recovered, 5)
    print("Pasos:", len(history) - 1)
    print("Energia inicial:", network.energy(damaged_x))
    print("Energia final:", network.energy(recovered))


if __name__ == "__main__":
    main()
