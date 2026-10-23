
#pragma once

#include "graphics_internal.hpp"

namespace application {

bool initialize();
void shutdown();

void update(double time);
void render(const graphics::internal::FrameData& fd);

void processInput(
    float deltaTime,
    bool forward,
    bool backward,
    bool left,
    bool right,
    bool up,
    bool down,
    bool fast
);

void rotateCamera(float deltaX, float deltaY);

} // namespace application
