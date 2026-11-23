
#version 450

// Теневой конвейер: единственный вершинный шейдер, который просто
// проецирует геометрию в пространство источника света. Фрагментная
// стадия отсутствует: используется динамический рендеринг
// (dynamic rendering, Vulkan 1.3), поэтому конвейер без фрагментного
// шейдера допустим для отрисовки только в карту глубины.
layout(location = 0) in vec3 inPosition;

// Полная матрица "проекция источника света * вид * модель"
// вычисляется на CPU и передаётся перед каждой командой рисования.
layout(push_constant) uniform ShadowTransform {
    mat4 lightModelViewProjection;
} shadow;

void main()
{
    gl_Position =
        shadow.lightModelViewProjection * vec4(inPosition, 1.0);
}
