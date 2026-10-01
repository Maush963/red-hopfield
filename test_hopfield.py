import unittest

from hopfield import HopfieldNetwork


class HopfieldNetworkTests(unittest.TestCase):
    def setUp(self):
        self.pattern_x = [
            1, -1, -1, -1, 1,
            -1, 1, -1, 1, -1,
            -1, -1, 1, -1, -1,
            -1, 1, -1, 1, -1,
            1, -1, -1, -1, 1,
        ]
        self.pattern_o = [
            -1, 1, 1, 1, -1,
            1, -1, -1, -1, 1,
            1, -1, -1, -1, 1,
            1, -1, -1, -1, 1,
            -1, 1, 1, 1, -1,
        ]
        self.network = HopfieldNetwork(25)
        self.network.train([self.pattern_x, self.pattern_o])

    def test_recall_returns_a_stored_pattern(self):
        damaged = self.pattern_x[:]
        damaged[6] = -1
        recovered = self.network.recall(damaged)[-1]
        self.assertEqual(recovered, self.pattern_x)

    def test_energy_does_not_increase_during_recall(self):
        damaged = self.pattern_x[:]
        damaged[6] = -1
        energies = []
        for state in self.network.recall(damaged):
            energies.append(self.network.energy(state))

        index = 1
        while index < len(energies):
            self.assertLessEqual(energies[index], energies[index - 1])
            index += 1

    def test_diagonal_weights_are_zero(self):
        row = 0
        while row < self.network.size:
            self.assertEqual(self.network.weights[row][row], 0)
            row += 1

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            self.network.recall([0] * 25)


if __name__ == "__main__":
    unittest.main()
