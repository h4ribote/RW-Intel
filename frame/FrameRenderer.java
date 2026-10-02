import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import org.newdawn.slick.opengl.renderer.SGL;

/**
 * Slick's renderer interface put in front of the real renderer, either dropping what draws or passing it through, and optionally carrying the frame hooks.
 *
 * Dropped are the calls that put pixels on the screen or only matter to calls that do: geometry, colour, matrices, display list playback and clearing.
 * Texture creation and binding, display list creation and every query go through, because loading a map builds textures and the game reads some of that state back.
 * The container calls glLoadIdentity between the update and the render of every frame and flush just before it paces the frame, which is where the instance in GameContainer.GL calls into {@link Frame}.
 */
final class FrameRenderer implements SGL {

    final SGL real;
    private final boolean drop;
    private final boolean hook;

    FrameRenderer(SGL real, boolean drop, boolean hook) {
        this.real = real;
        this.drop = drop;
        this.hook = hook;
    }

    // The frame hooks.

    public void glLoadIdentity() {
        if (hook) Frame.beforeRender();
        if (!drop) real.glLoadIdentity();
    }

    public void flush() {
        if (hook) Frame.beforeSync();
        if (!drop) real.flush();
    }

    // Drawing, dropped when drawing is off.

    public void glClear(int mask) {
        if (!drop) real.glClear(mask);
    }

    public void glLineWidth(float width) {
        if (!drop) real.glLineWidth(width);
    }

    public void glPointSize(float size) {
        if (!drop) real.glPointSize(size);
    }

    public void glColor4f(float r, float g, float b, float a) {
        if (!drop) real.glColor4f(r, g, b, a);
    }

    public void glSecondaryColor3ubEXT(byte r, byte g, byte b) {
        if (!drop) real.glSecondaryColor3ubEXT(r, g, b);
    }

    public void glTexCoord2f(float u, float v) {
        if (!drop) real.glTexCoord2f(u, v);
    }

    public void glVertex3f(float x, float y, float z) {
        if (!drop) real.glVertex3f(x, y, z);
    }

    public void glVertex2f(float x, float y) {
        if (!drop) real.glVertex2f(x, y);
    }

    public void glBegin(int geomType) {
        if (!drop) real.glBegin(geomType);
    }

    public void glEnd() {
        if (!drop) real.glEnd();
    }

    public void glCallList(int id) {
        if (!drop) real.glCallList(id);
    }

    public void glRotatef(float angle, float x, float y, float z) {
        if (!drop) real.glRotatef(angle, x, y, z);
    }

    public void glTranslatef(float x, float y, float z) {
        if (!drop) real.glTranslatef(x, y, z);
    }

    public void glScalef(float x, float y, float z) {
        if (!drop) real.glScalef(x, y, z);
    }

    public void glPushMatrix() {
        if (!drop) real.glPushMatrix();
    }

    public void glPopMatrix() {
        if (!drop) real.glPopMatrix();
    }

    public void glLoadMatrix(FloatBuffer buffer) {
        if (!drop) real.glLoadMatrix(buffer);
    }

    // State, textures and queries, always passed through.

    public void initDisplay(int width, int height) {
        real.initDisplay(width, height);
    }

    public void enterOrtho(int width, int height) {
        real.enterOrtho(width, height);
    }

    public void glClearColor(float r, float g, float b, float a) {
        real.glClearColor(r, g, b, a);
    }

    public void glClipPlane(int plane, DoubleBuffer buffer) {
        real.glClipPlane(plane, buffer);
    }

    public void glScissor(int x, int y, int width, int height) {
        real.glScissor(x, y, width, height);
    }

    public void glColorMask(boolean red, boolean green, boolean blue, boolean alpha) {
        real.glColorMask(red, green, blue, alpha);
    }

    public void glGetInteger(int id, IntBuffer result) {
        real.glGetInteger(id, result);
    }

    public void glGetFloat(int id, FloatBuffer result) {
        real.glGetFloat(id, result);
    }

    public void glEnable(int item) {
        real.glEnable(item);
    }

    public void glDisable(int item) {
        real.glDisable(item);
    }

    public void glBindTexture(int target, int id) {
        real.glBindTexture(target, id);
    }

    public void glGetTexImage(int target, int level, int format, int type, ByteBuffer pixels) {
        real.glGetTexImage(target, level, format, type, pixels);
    }

    public void glDeleteTextures(IntBuffer buffer) {
        real.glDeleteTextures(buffer);
    }

    public void glTexEnvi(int target, int mode, int value) {
        real.glTexEnvi(target, mode, value);
    }

    public void glBlendFunc(int src, int dest) {
        real.glBlendFunc(src, dest);
    }

    public int glGenLists(int count) {
        return real.glGenLists(count);
    }

    public void glNewList(int id, int option) {
        real.glNewList(id, option);
    }

    public void glEndList() {
        real.glEndList();
    }

    public void glCopyTexImage2D(int target, int level, int internalFormat, int x, int y, int width, int height, int border) {
        real.glCopyTexImage2D(target, level, internalFormat, x, y, width, height, border);
    }

    public void glReadPixels(int x, int y, int width, int height, int format, int type, ByteBuffer pixels) {
        real.glReadPixels(x, y, width, height, format, type, pixels);
    }

    public void glTexParameteri(int target, int param, int value) {
        real.glTexParameteri(target, param, value);
    }

    public float[] getCurrentColor() {
        return real.getCurrentColor();
    }

    public void glDeleteLists(int list, int count) {
        real.glDeleteLists(list, count);
    }

    public void glDepthMask(boolean mask) {
        real.glDepthMask(mask);
    }

    public void glClearDepth(float value) {
        real.glClearDepth(value);
    }

    public void glDepthFunc(int func) {
        real.glDepthFunc(func);
    }

    public void setGlobalAlphaScale(float alphaScale) {
        real.setGlobalAlphaScale(alphaScale);
    }

    public void glGenTextures(IntBuffer ids) {
        real.glGenTextures(ids);
    }

    public void glGetError() {
        real.glGetError();
    }

    public void glTexImage2D(int target, int i, int dstPixelFormat, int width, int height, int j, int srcPixelFormat, int glUnsignedByte, ByteBuffer textureBuffer) {
        real.glTexImage2D(target, i, dstPixelFormat, width, height, j, srcPixelFormat, glUnsignedByte, textureBuffer);
    }

    public void glTexSubImage2D(int glTexture2d, int i, int pageX, int pageY, int width, int height, int glBgra, int glUnsignedByte, ByteBuffer scratchByteBuffer) {
        real.glTexSubImage2D(glTexture2d, i, pageX, pageY, width, height, glBgra, glUnsignedByte, scratchByteBuffer);
    }

    public boolean canTextureMirrorClamp() {
        return real.canTextureMirrorClamp();
    }

    public boolean canSecondaryColor() {
        return real.canSecondaryColor();
    }
}
