#!/bin/bash
# Guardrail script to check for invalid C++ float literals in actor.h
# Run this after generating any checkpoint to prevent build failures
#
# Invalid pattern: integer immediately followed by 'f' (e.g., "10f", "245760f")
# Valid pattern: decimal number followed by 'f' (e.g., "10.0f", "245760.0f")

ACTOR_FILE="${1:-controller/actor.h}"

if [[ ! -f "$ACTOR_FILE" ]]; then
    echo "Error: $ACTOR_FILE not found"
    exit 1
fi

# Search for invalid float literals: = followed by digits then 'f' without a decimal point
# Pattern: = [digits]f (no '.' before the 'f')
INVALID_LITERALS=$(grep -nE '= [0-9]+f[^0-9]' "$ACTOR_FILE")

if [[ -n "$INVALID_LITERALS" ]]; then
    echo "ERROR: Invalid float literals found in $ACTOR_FILE"
    echo "These will cause 'unable to find numeric literal operator' errors during firmware build."
    echo ""
    echo "Found issues:"
    echo "$INVALID_LITERALS"
    echo ""
    echo "Fix: Change integer literals like '10f' to '10.0f'"
    exit 1
fi

echo "OK: No invalid float literals found in $ACTOR_FILE"
exit 0
