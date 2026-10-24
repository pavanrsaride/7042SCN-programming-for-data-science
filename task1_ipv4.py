"""7042SCN Task 1 — IPv4 to dotted 32-bit binary (no imported libraries).

The assignment's function signature and two original assertions are retained.
"""


def convert_ip(ip_address):
    if not isinstance(ip_address, str):
        raise TypeError("An IPv4 address must be supplied as a string.")

    if ip_address == "127.0.0.1":
        return "There is no place like localhost"

    octets = ip_address.split(".")
    if len(octets) != 4:
        raise ValueError("An IPv4 address must have exactly four octets.")

    binary_octets = []
    for octet in octets:
        if not octet or any(digit not in "0123456789" for digit in octet):
            raise ValueError("Each octet must contain decimal digits only.")

        number = int(octet)
        if not 0 <= number <= 255:
            raise ValueError("IPv4 octets must be between 0 and 255.")

        binary_octets.append(f"{number:08b}")

    return ".".join(binary_octets)


# Unchanged assertions required by the assignment brief.
assert convert_ip("192.168.1.1") == "11000000.10101000.00000001.00000001"
assert convert_ip("127.0.0.1") == "There is no place like localhost"


def run_demonstration():
    print("TASK 1 — IPV4 CONVERSION")
    for address in ("192.168.1.1", "127.0.0.1", "0.0.0.0", "255.255.255.255"):
        print(f"{address} -> {convert_ip(address)}")
    print("Both supplied assertions passed.")


if __name__ == "__main__":
    run_demonstration()
