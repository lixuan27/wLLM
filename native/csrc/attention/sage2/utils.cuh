/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0.
 *
 * wLLM note: this local header intentionally drops SageAttention's
 * torch::Tensor validation macros. The raw-pointer wLLM binding validates
 * shape/layout at the call boundary and launches only the Motus SM120 path.
 */

#pragma once
