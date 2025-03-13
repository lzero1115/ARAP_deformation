#include <igl/read_triangle_mesh.h>
#include <Eigen/Core>

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>

#include <vector>
#include <limits>
#include <iostream>
#include <algorithm>
#include <string>
#include <cmath>

// Global mesh variables
Eigen::MatrixXd V;  // Vertex positions (working/temporary copy)
Eigen::MatrixXd V_orig;
Eigen::MatrixXi F;  // Face indices

std::vector<int> landmarks;
float vertexRadius = 0.004f; // For displaying landmark points

static glm::mat4 savedViewMatrix;

static bool cameraLocked = false;


// Interaction state
static bool allowSelection = false;     // When true, clicking selects/deselects vertices.
static bool allowInteraction = false;   // When true, selection is disabled and dragging is enabled.
static int activeVertex = -1;           // Index of the vertex being dragged (if any).

// Helper: Convert a world-space position to screen coordinates.
glm::vec2 worldToScreen(const glm::vec3& worldPos) {
    glm::mat4 view = polyscope::view::getCameraViewMatrix();
    glm::mat4 proj = polyscope::view::getCameraPerspectiveMatrix();
    glm::vec4 clip = proj * view * glm::vec4(worldPos, 1.0f);
    glm::vec3 ndc = glm::vec3(clip) / clip.w;
    float x = (ndc.x + 1.0f) * 0.5f * ImGui::GetIO().DisplaySize.x;
    float y = (1.0f - ndc.y) * 0.5f * ImGui::GetIO().DisplaySize.y;
    return glm::vec2(x, y);
}

// Unproject a mouse position to obtain a ray (origin and normalized direction)
std::pair<glm::vec3, glm::vec3> unproject(const ImVec2& mousePos) {
    glm::mat4 projMatrix = polyscope::view::getCameraPerspectiveMatrix();
    glm::mat4 viewMatrix = polyscope::view::getCameraViewMatrix();
    glm::mat4 invProj = glm::inverse(projMatrix);
    glm::mat4 invView = glm::inverse(viewMatrix);

    float x = (2.0f * mousePos.x) / ImGui::GetIO().DisplaySize.x - 1.0f;
    float y = 1.0f - (2.0f * mousePos.y) / ImGui::GetIO().DisplaySize.y;

    glm::vec4 rayClip(x, y, -1.0f, 1.0f);
    glm::vec4 rayEye = invProj * rayClip;
    rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);
    glm::vec4 rayWorld = invView * rayEye;

    glm::vec3 cameraPos = polyscope::view::getCameraWorldPosition();
    glm::vec3 rayDir = glm::normalize(glm::vec3(rayWorld));
    return {cameraPos, rayDir};
}

// Möller–Trumbore ray–triangle intersection.
// Returns true if the ray (starting at orig in direction dir) intersects the triangle (v0, v1, v2),
// and sets t to the distance along the ray.
bool rayTriangleIntersect(
    const glm::vec3& orig,
    const glm::vec3& dir,
    const glm::vec3& v0,
    const glm::vec3& v1,
    const glm::vec3& v2,
    float &t)
{
    const float EPSILON = 1e-8f;
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 h = glm::cross(dir, edge2);
    float a = glm::dot(edge1, h);
    if (a > -EPSILON && a < EPSILON)
        return false; // Parallel ray.
    float f = 1.0f / a;
    glm::vec3 s = orig - v0;
    float u = f * glm::dot(s, h);
    if (u < 0.0f || u > 1.0f)
        return false;
    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(dir, q);
    if (v < 0.0f || u + v > 1.0f)
        return false;
    t = f * glm::dot(edge2, q);
    return (t > EPSILON);
}

// Update the display of landmark points.
void updateLandmarksDisplay() {
    std::vector<glm::vec3> landmarkPoints;
    for (int idx : landmarks) {
        landmarkPoints.push_back(glm::vec3(V(idx, 0), V(idx, 1), V(idx, 2)));
    }
    if (polyscope::hasPointCloud("Landmarks"))
        polyscope::removePointCloud("Landmarks");
    polyscope::registerPointCloud("Landmarks", landmarkPoints)
        ->setPointRadius(vertexRadius)
        ->setPointColor(glm::vec3(1.0f, 0.65f, 0.0f));
}

// In selection mode, given a mouse click, find the intersected triangle and toggle the closest vertex.
void selectNearestVertex(const ImVec2& mousePos) {
    auto [rayOrigin, rayDir] = unproject(mousePos);
    float bestT = std::numeric_limits<float>::max();
    bool hit = false;
    glm::vec3 bestIntersection;

    // Iterate over every face (triangle) in the mesh.
    for (int i = 0; i < F.rows(); i++) {
        int i0 = F(i, 0);
        int i1 = F(i, 1);
        int i2 = F(i, 2);
        glm::vec3 v0(V(i0, 0), V(i0, 1), V(i0, 2));
        glm::vec3 v1(V(i1, 0), V(i1, 1), V(i1, 2));
        glm::vec3 v2(V(i2, 0), V(i2, 1), V(i2, 2));
        float t;
        if (rayTriangleIntersect(rayOrigin, rayDir, v0, v1, v2, t)) {
            if (t < bestT) {
                bestT = t;
                bestIntersection = rayOrigin + rayDir * t;
                hit = true;
            }
        }
    }
    if (hit) {
        float minDist = std::numeric_limits<float>::max();
        int closestVertex = -1;
        for (int i = 0; i < V.rows(); i++) {
            glm::vec3 vertex(V(i, 0), V(i, 1), V(i, 2));
            float dist = glm::length(vertex - bestIntersection);
            if (dist < minDist) {
                minDist = dist;
                closestVertex = i;
            }
        }
        if (closestVertex != -1) {
            auto it = std::find(landmarks.begin(), landmarks.end(), closestVertex);
            if (it == landmarks.end())
                landmarks.push_back(closestVertex);
            else
                landmarks.erase(it);
            updateLandmarksDisplay();
        }
    }
}

// User callback: draws the UI and handles mouse input.
void userCallback() {
    ImGui::PushItemWidth(100);

    if (ImGui::Checkbox("Allow Selection", &allowSelection)) {
        if (allowSelection) {
            // If enabling selection, disable interaction
            allowInteraction = false;
            cameraLocked = false;
        }
    }


    // When Interaction checkbox is clicked
    if (ImGui::Checkbox("Interaction Mode", &allowInteraction)) {
        if (allowInteraction) {
            // If enabling interaction, disable selection and lock camera
            allowSelection = false;
            savedViewMatrix = polyscope::view::getCameraViewMatrix();
            cameraLocked = true;
        } else {
            // Release camera lock when exiting interaction mode
            cameraLocked = false;
        }
    }

    // If camera is locked, restore the saved camera parameters every frame
    if (cameraLocked) {
        polyscope::view::setCameraViewMatrix(savedViewMatrix);
    }

    if (ImGui::Button("Reset")) {
        landmarks.clear();
        if (polyscope::hasPointCloud("Landmarks"))
            polyscope::removePointCloud("Landmarks");

        V = V_orig;
        polyscope::getSurfaceMesh("Mesh")->updateVertexPositions(V);
        polyscope::requestRedraw();
    }
    ImGui::End();
    // --- Interaction Mode ---
    if (allowInteraction) {

        // When in interaction mode, disable selection.
        // On mouse click, if no vertex is already active, check if a landmark is clicked.
        if (ImGui::IsMouseClicked(0) && !ImGui::GetIO().WantCaptureMouse && activeVertex == -1) {
            ImVec2 mousePos = ImGui::GetMousePos();
            float minScreenDist = 20.0f; // Pixel threshold.
            int candidate = -1;
            for (int idx : landmarks) {
                glm::vec3 pos(V(idx, 0), V(idx, 1), V(idx, 2));
                glm::vec2 screenPos = worldToScreen(pos);
                glm::vec2 diff = glm::vec2(mousePos.x, mousePos.y) - screenPos;
                float dist = glm::length(diff);
                if (dist < minScreenDist) {
                    minScreenDist = dist;
                    candidate = idx;
                }
            }
            if (candidate != -1)
                activeVertex = candidate;
        }

        // While dragging, update the active vertex's position.
        if (activeVertex != -1 && ImGui::IsMouseDragging(0) && !ImGui::GetIO().WantCaptureMouse) {
            // Define a drag plane parallel to the view (normal equals view direction).
            auto center = ImVec2(ImGui::GetIO().DisplaySize.x / 2.0f,
                                 ImGui::GetIO().DisplaySize.y / 2.0f);
            auto [dummyOrigin, viewDir] = unproject(center);
            glm::vec3 normal = viewDir; // Plane normal.
            // Get the current ray from the mouse.
            ImVec2 mousePos = ImGui::GetMousePos();
            auto [rayOrigin, rayDir] = unproject(mousePos);
            // Define the drag plane passing through the active vertex.
            glm::vec3 planePoint(V(activeVertex, 0), V(activeVertex, 1), V(activeVertex, 2));
            float denom = glm::dot(rayDir, normal);
            if (std::fabs(denom) > 1e-6) {
                float t = glm::dot(planePoint - rayOrigin, normal) / denom;
                glm::vec3 newPos = rayOrigin + t * rayDir;
                V(activeVertex, 0) = newPos.x;
                V(activeVertex, 1) = newPos.y;
                V(activeVertex, 2) = newPos.z;
                polyscope::getSurfaceMesh("Mesh")->updateVertexPositions(V);
                updateLandmarksDisplay();
            }

            // Ensure camera stays locked during drag operations
            if (cameraLocked) {
                polyscope::view::setCameraViewMatrix(savedViewMatrix);
            }
        }

        // On mouse release, stop dragging.
        if (ImGui::IsMouseReleased(0))
            activeVertex = -1;
    }
    // --- Selection Mode ---
    else if (allowSelection) {

        if (ImGui::IsMouseClicked(0) && !ImGui::GetIO().WantCaptureMouse) {
            ImVec2 mousePos = ImGui::GetMousePos();
            selectNearestVertex(mousePos);
        }
    }
}

int main(int argc, char** argv) {
    std::string filename = (argc > 1) ? std::string(argv[1]) : "../../data/bunny.off";
    if (!igl::read_triangle_mesh(filename, V, F)) {
        std::cerr << "Failed to load mesh: " << filename << std::endl;
        return EXIT_FAILURE;
    }
    V_orig = V;

    polyscope::init();
    polyscope::registerSurfaceMesh("Mesh", V, F);
    polyscope::state::userCallback = userCallback;
    polyscope::show();

    return EXIT_SUCCESS;
}
